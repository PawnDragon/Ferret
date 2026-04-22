import gc

import numpy as np
from tqdm import tqdm
from copy import deepcopy

from optimizers.ferret_optimizer import *
from optimizers.lora_utils import (
    build_lora_model,
    extract_classifier_state,
    extract_lora_A_state,
    extract_lora_B_state,
    extract_lora_state,
    load_classifier_state,
    load_lora_A_state,
    load_lora_B_state,
    load_lora_state,
)
from optimizers.florg_utils import (
    build_florg_model,
    extract_florg_A_state,
    load_florg_A_state,
)


class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.device}')
        else:
            self.device = torch.device('cpu')
        self.candidate_seeds = candidate_seeds
        self._optim_debug_logged = False
        self._fedex_optim_reset_logged = False
        self._florg_shape_logged = False
        self.local_lora_B_state = None
        self.prev_round_lora_A_state = None

    def _maybe_log_florg_shapes(self, cur_round):
        if self._florg_shape_logged or self.idx != 0:
            return
        for layer_name, module in self.model.named_modules():
            if not hasattr(module, 'florg_A'):
                continue
            print(
                f'[debug][florg][client{self.idx}] round {cur_round} '
                f'layer={layer_name} L_shape={tuple(module.florg_L.shape)} '
                f'R_shape={tuple(module.florg_R.shape)} A_shape={tuple(module.florg_A.shape)}'
            )
            self._florg_shape_logged = True
            return

    def _clear_framework_optimizer_state(self, framework):
        if framework is None:
            return
        optim = getattr(framework, 'optim', None)
        if optim is None:
            return
        try:
            optim.zero_grad(set_to_none=True)
        except TypeError:
            optim.zero_grad()
        if hasattr(optim, 'state') and isinstance(optim.state, dict):
            optim.state.clear()
        framework.optim = None

    def _release_florg_training_state(self, framework):
        self._clear_framework_optimizer_state(framework)
        if self.model is not None:
            try:
                self.model.zero_grad(set_to_none=True)
            except TypeError:
                self.model.zero_grad()
            self.model.to('cpu')
        self.model = None
        if framework is not None:
            framework.model = None
        del framework
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_runtime_debug_context(self, cur_round):
        # Shared args object carries lightweight runtime context for framework-level debug prints.
        setattr(self.args, '_runtime_client_idx', int(self.idx))
        setattr(self.args, '_runtime_round_idx', int(cur_round))

    def _next_batch(self):
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'labels': batch['labels'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }

    def _maybe_log_loaded_named_state(self, framework, named_state, cur_round, tag):
        if self._optim_debug_logged or self.idx != 0:
            return
        if not isinstance(named_state, dict):
            return
        state_dict = named_state.get('state', {})
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            return
        sampled_name = sorted(state_dict.keys())[0]
        param_obj = dict(framework.model.named_parameters()).get(sampled_name, None)
        if param_obj is None:
            return
        opt_state = framework.optim.state.get(param_obj, {})
        exp_avg = opt_state.get('exp_avg', None)
        exp_avg_sq = opt_state.get('exp_avg_sq', None)
        if isinstance(exp_avg, torch.Tensor) and isinstance(exp_avg_sq, torch.Tensor):
            print(
                f'[debug][client{self.idx}][{tag}] loaded optim state @ {sampled_name}: '
                f'exp_avg.norm={float(exp_avg.norm().item()):.6e}, '
                f'exp_avg_sq.norm={float(exp_avg_sq.norm().item()):.6e}'
            )
            self._optim_debug_logged = True

    def _state_l2_norm(self, state_dict):
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            return 0.0
        total = 0.0
        for tensor in state_dict.values():
            if not isinstance(tensor, torch.Tensor):
                continue
            total += float(torch.norm(tensor.float()).item())
        return total

    def _state_delta_l2_norm(self, state_a, state_b):
        if not isinstance(state_a, dict) or not isinstance(state_b, dict):
            return float('nan')
        keys = sorted(set(state_a.keys()).intersection(set(state_b.keys())))
        if len(keys) == 0:
            return float('nan')
        total = 0.0
        for key in keys:
            tensor_a = state_a[key]
            tensor_b = state_b[key]
            if (not isinstance(tensor_a, torch.Tensor)) or (not isinstance(tensor_b, torch.Tensor)):
                continue
            total += float(torch.norm(tensor_a.float() - tensor_b.float()).item())
        return total

    def _prepare_fedsubmuonv2_round_state(self, submuon_state):
        if not isinstance(submuon_state, dict):
            raise RuntimeError('[fedsubmuonv2] missing submuon_state in client local train')
        if 'x_global' not in submuon_state or 'seeds' not in submuon_state:
            raise RuntimeError('[fedsubmuonv2] submuon_state must contain x_global and seeds')
        x_global = submuon_state['x_global']
        seeds = submuon_state['seeds']
        if not isinstance(x_global, dict) or not isinstance(seeds, dict):
            raise RuntimeError('[fedsubmuonv2] invalid x_global or seeds type')
        return x_global, None, seeds

    def _compute_fedsubmuon_gt_probe(self, framework, probe_batches):
        probe_batches = int(max(probe_batches, 0))
        if probe_batches <= 0:
            return {}, {}
        if framework is None or self.model is None:
            return {}, {}

        target_layers = list(framework.submuon_x.keys())
        if len(target_layers) == 0:
            return {}, {}

        module_map = dict(self.model.named_modules())
        target_modules = {}
        old_requires_grad = {}
        h_u = {}
        h_v = {}

        for layer_name in target_layers:
            module = module_map.get(layer_name, None)
            if module is None or (not hasattr(module, 'weight')):
                continue
            target_modules[layer_name] = module
            old_requires_grad[layer_name] = bool(module.weight.requires_grad)
            module.weight.requires_grad_(True)

        if len(target_modules) == 0:
            return {}, {}

        for layer_name, module in target_modules.items():
            U, V = framework.get_submuon_uv(
                layer_name=layer_name,
                out_dim=int(module.out_features),
                in_dim=int(module.in_features),
                device=module.weight.device,
                dtype=torch.float32,
            )
            h_u[layer_name] = torch.zeros_like(U, dtype=torch.float32, device=module.weight.device)
            h_v[layer_name] = torch.zeros_like(V, dtype=torch.float32, device=module.weight.device)

        used_batches = 0
        for _ in range(probe_batches):
            batch = self._next_batch()
            try:
                self.model.zero_grad(set_to_none=True)
            except TypeError:
                self.model.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            if not torch.isfinite(loss):
                continue
            loss.backward()
            used_batches += 1
            with torch.no_grad():
                for layer_name, module in target_modules.items():
                    grad_w = module.weight.grad
                    if grad_w is None:
                        continue
                    U, V = framework.get_submuon_uv(
                        layer_name=layer_name,
                        out_dim=int(module.out_features),
                        in_dim=int(module.in_features),
                        device=grad_w.device,
                        dtype=torch.float32,
                    )
                    h_u[layer_name].add_(grad_w.detach().float() @ V.float())
                    h_v[layer_name].add_(grad_w.detach().float().t() @ U.float())

        denom = float(max(used_batches, 1))
        for layer_name in h_u.keys():
            h_u[layer_name].div_(denom)
            h_v[layer_name].div_(denom)

        try:
            self.model.zero_grad(set_to_none=True)
        except TypeError:
            self.model.zero_grad()
        for x_param in framework.submuon_x.values():
            if isinstance(x_param, torch.nn.Parameter):
                x_param.grad = None
        for layer_name, module in target_modules.items():
            module.weight.requires_grad_(old_requires_grad[layer_name])

        h_u_out = {k: v.detach().cpu().contiguous() for k, v in h_u.items()}
        h_v_out = {k: v.detach().cpu().contiguous() for k, v in h_v.items()}
        return h_u_out, h_v_out

    def local_train_with_seed_pool(
        self,
        pulled_model,
        cur_round,
        submuon_state=None,
        krso_state=None,
        multisub_state=None,
        struct_state=None,
        lora_state=None,
        lora_A_state=None,
        lora_B_state=None,
        classifier_state=None,
        florg_A_state=None,
        florg_seed_state=None,
        florg_basis_state=None,
        fedavg_global_state=None,
        global_named_optim_state=None,
    ):
        self.model = pulled_model
        self.model.to(self.device)

        if getattr(self.args, 'algo', 'ferret') == 'fedmultisubmuon':
            if not isinstance(multisub_state, dict):
                raise RuntimeError('[fedmultisubmuon] missing multisub_state in client local train')
            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_multisub_state(multisub_state=multisub_state, trainable=True)
            self.model.train()

            loss_total_train = 0.0
            num_trained = 0

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
                progress_bar = tqdm(range(iter_steps))
                for cur_step, batch in enumerate(self.train_loader):
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
            else:
                iter_steps = max(int(self.args.local_step), 1)
                progress_bar = tqdm(range(iter_steps))
                for cur_step in range(iter_steps):
                    batch = self._next_batch()
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )

            b_local, c_local, score_local = framework.export_multisub_state()
            framework.clear_multisub_state()
            self.model = None
            return {
                'b': b_local,
                'c': c_local,
                'scores': score_local,
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        if getattr(self.args, 'algo', 'ferret') == 'fedstructmuon':
            if not isinstance(struct_state, dict):
                raise RuntimeError('[fedstructmuon] missing struct_state in client local train')
            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_struct_state(struct_state=struct_state, trainable=True)
            self.model.train()

            loss_total_train = 0.0
            num_trained = 0

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
                progress_bar = tqdm(range(iter_steps))
                for cur_step, batch in enumerate(self.train_loader):
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
            else:
                iter_steps = max(int(self.args.local_step), 1)
                progress_bar = tqdm(range(iter_steps))
                for cur_step in range(iter_steps):
                    batch = self._next_batch()
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )

            x_local, score_local = framework.export_struct_state()
            framework.clear_struct_state()
            self.model = None
            return {
                'x': x_local,
                'scores': score_local,
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        if getattr(self.args, 'algo', 'ferret') in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd']:
            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            algo_name = getattr(self.args, 'algo', 'ferret')
            uv_state = None
            is_refresh_round = False
            is_probe_client = False
            if algo_name == 'fedsubmuonv2':
                x_state, m_state, seeds = self._prepare_fedsubmuonv2_round_state(submuon_state)
            elif algo_name == 'fedsubmuon_gt':
                if not isinstance(submuon_state, dict):
                    raise RuntimeError('[fedsubmuon_gt] missing submuon_state in client local train')
                if 'x_global' not in submuon_state or 'seeds' not in submuon_state:
                    raise RuntimeError('[fedsubmuon_gt] submuon_state must contain x_global and seeds')
                x_state = submuon_state['x_global']
                m_state = None
                seeds = submuon_state['seeds']
                u_state = submuon_state.get('u_global', None)
                v_state_basis = submuon_state.get('v_basis_global', None)
                if isinstance(u_state, dict) and isinstance(v_state_basis, dict):
                    uv_state = {'u': u_state, 'v': v_state_basis}
                is_refresh_round = bool(submuon_state.get('is_refresh_round', False))
                is_probe_client = bool(submuon_state.get('is_probe_client', is_refresh_round))
            else:
                x_state = submuon_state['x_global']
                m_state = submuon_state.get('m_global', None)
                seeds = submuon_state['seeds']
            framework.set_submuon_state(
                x_state=x_state,
                m_state=m_state,
                seeds=seeds,
                trainable=True,
                v_state=submuon_state.get('v_global', None),
                uv_state=uv_state,
            )
            self.model.train()

            loss_total_train = 0.0
            num_trained = 0

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
                progress_bar = tqdm(range(iter_steps))
                for cur_step, batch in enumerate(self.train_loader):
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
            else:
                iter_steps = max(int(self.args.local_step), 1)
                progress_bar = tqdm(range(iter_steps))
                for cur_step in range(iter_steps):
                    batch = self._next_batch()
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )

            aggregate_muon_state = should_aggregate_submuon_m_state(self.args, algo_name)
            h_u_local = None
            h_v_local = None
            if algo_name == 'fedsubmuon_gt' and is_refresh_round and is_probe_client:
                h_u_local, h_v_local = self._compute_fedsubmuon_gt_probe(
                    framework=framework,
                    probe_batches=int(getattr(self.args, 'gt_probe_batches', 1)),
                )

            if algo_name == 'fedsubadam':
                x_local = framework.export_submuon_state(with_m=False)
            elif algo_name == 'fedsubsgd':
                x_local = framework.export_submuon_state(with_m=False)
            elif algo_name == 'fedsubmuon' and aggregate_muon_state:
                x_local, m_local = framework.export_submuon_state()
            else:
                x_local = framework.export_submuon_state(with_m=False)
            framework.clear_submuon_state()
            self.model = None
            payload = {'x': x_local, 'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0}
            if algo_name == 'fedsubmuon' and aggregate_muon_state:
                payload['m'] = m_local
            if algo_name == 'fedsubmuon_gt' and is_refresh_round:
                payload['is_probe_client'] = int(bool(is_probe_client))
                payload['h_u'] = h_u_local if isinstance(h_u_local, dict) else {}
                payload['h_v'] = h_v_local if isinstance(h_v_local, dict) else {}
            return payload

        if getattr(self.args, 'algo', 'ferret') == 'fedkrso':
            if not isinstance(krso_state, dict):
                raise RuntimeError('[fedkrso] missing krso_state in client local train')
            seed_pool = krso_state.get('seed_pool', [])
            if not isinstance(seed_pool, (list, tuple)) or len(seed_pool) == 0:
                raise RuntimeError('[fedkrso] krso_state must contain non-empty seed_pool')

            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_krso_state(krso_state=krso_state, trainable=True)
            self.model.train()

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
                loader_iter = iter(self.train_loader)
            else:
                iter_steps = max(int(self.args.local_step), 1)
                loader_iter = None
            interval_len = max(int(getattr(self.args, 'krso_interval_len', 10)), 1)

            loss_total_train = 0.0
            num_trained = 0
            interval_delta_norm_sum = 0.0
            progress_bar = tqdm(range(iter_steps))
            cur_step = 0
            while cur_step < iter_steps:
                active_seed = int(np.random.choice(seed_pool))
                framework.begin_krso_interval(active_seed)
                remaining = int(iter_steps - cur_step)
                interval_steps = int(min(interval_len, remaining))
                for interval_step in range(interval_steps):
                    if loader_iter is not None:
                        batch_raw = next(loader_iter)
                        batch = {
                            'input_ids': batch_raw['input_ids'].to(self.device),
                            'labels': batch_raw['labels'].to(self.device),
                            'attention_mask': batch_raw['attention_mask'].to(self.device),
                        }
                    else:
                        batch = self._next_batch()
                    apply_optim_step = (
                        (interval_step % self.args.n_accum == self.args.n_accum - 1)
                        or (interval_step == interval_steps - 1)
                    )
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: '
                        f'{loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
                    cur_step += 1
                interval_stats = framework.finish_krso_interval()
                interval_delta_norm_sum += float(interval_stats.get('delta_norm', 0.0))

            b_local, used_seeds = framework.export_krso_state()
            num_intervals = int(getattr(framework, 'krso_num_intervals', 0))
            framework.clear_krso_state()
            self.model = None
            return {
                'b': b_local,
                'used_seeds': list(used_seeds),
                'num_intervals': num_intervals,
                'num_active_seeds': int(len(used_seeds)),
                'interval_delta_norm': float(interval_delta_norm_sum),
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        if getattr(self.args, 'algo', 'ferret') in ['fedit', 'federa', 'flora', 'fedsalora', 'fedexlora', 'florg']:
            if getattr(self.args, 'algo', 'ferret') == 'florg':
                if florg_basis_state is None:
                    raise RuntimeError('[florg] missing basis_state in client local train')
                self.model = build_florg_model(
                    self.model,
                    self.args,
                    seed_state=florg_seed_state,
                    basis_state=florg_basis_state,
                )
            else:
                self.model = build_lora_model(self.model, self.args)
            self.model.to(self.device)
            if getattr(self.args, 'algo', 'ferret') in ['fedit', 'federa']:
                load_lora_state(self.model, lora_state)
            elif getattr(self.args, 'algo', 'ferret') == 'fedsalora':
                load_lora_A_state(self.model, lora_A_state)
                if self.local_lora_B_state is not None:
                    load_lora_B_state(self.model, self.local_lora_B_state)
                if self.idx == 0 and cur_round <= 2:
                    loaded_a_state = extract_lora_A_state(self.model)
                    loaded_b_state = extract_lora_B_state(self.model)
                    loaded_a_norm = self._state_l2_norm(loaded_a_state)
                    loaded_b_norm = self._state_l2_norm(loaded_b_state)
                    a_shift = self._state_delta_l2_norm(loaded_a_state, self.prev_round_lora_A_state)
                    b_recover_delta = self._state_delta_l2_norm(loaded_b_state, self.local_lora_B_state)
                    print(
                        f'[debug][fedsalora][client{self.idx}] round {cur_round} start: '
                        f'loaded_A_norm={loaded_a_norm:.6e}, loaded_B_norm={loaded_b_norm:.6e}, '
                        f'A_shift_from_prev={a_shift:.6e}, B_reload_delta={b_recover_delta:.6e}'
                    )
            elif getattr(self.args, 'algo', 'ferret') == 'fedexlora':
                load_lora_A_state(self.model, lora_A_state)
                load_lora_B_state(self.model, lora_B_state)
                load_classifier_state(self.model, classifier_state)
            elif getattr(self.args, 'algo', 'ferret') == 'florg':
                load_florg_A_state(self.model, florg_A_state)
                self._maybe_log_florg_shapes(cur_round)

            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            if (
                getattr(self.args, 'algo', 'ferret') == 'fedexlora'
                and int(cur_round) == 1
                and (not self._fedex_optim_reset_logged)
            ):
                print(
                    f'[debug][fedexlora][client{self.idx}] round {cur_round} '
                    f'optimizer_state_len_before_train={len(framework.optim.state)}'
                )
                self._fedex_optim_reset_logged = True
            self.model.train()
            self.model.zero_grad()

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
            else:
                iter_steps = max(int(self.args.local_step), 1)

            loss_total_train = 0.0
            num_trained = 0
            progress_bar = tqdm(range(iter_steps))

            if self.args.batch_or_epoch == 'epoch':
                for cur_step, batch in enumerate(self.train_loader):
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
            else:
                for cur_step in range(iter_steps):
                    batch = self._next_batch()
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )

            if getattr(self.args, 'algo', 'ferret') == 'fedsalora':
                local_lora_A_state = extract_lora_A_state(self.model)
                local_lora_B_state = extract_lora_B_state(self.model)
                if self.idx == 0 and cur_round <= 2:
                    end_a_norm = self._state_l2_norm(local_lora_A_state)
                    end_b_norm = self._state_l2_norm(local_lora_B_state)
                    prev_b_delta = self._state_delta_l2_norm(local_lora_B_state, self.local_lora_B_state)
                    print(
                        f'[debug][fedsalora][client{self.idx}] round {cur_round} end: '
                        f'A_norm={end_a_norm:.6e}, B_norm={end_b_norm:.6e}, '
                        f'B_delta_from_prev={prev_b_delta:.6e}'
                    )
                self.prev_round_lora_A_state = {k: v.clone() for k, v in local_lora_A_state.items()}
                self.local_lora_B_state = {k: v.clone() for k, v in local_lora_B_state.items()}
                self.model = None
                return {
                    'lora_A_state': local_lora_A_state,
                    'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
                }
            if getattr(self.args, 'algo', 'ferret') == 'fedexlora':
                local_lora_A_state = extract_lora_A_state(self.model)
                local_lora_B_state = extract_lora_B_state(self.model)
                local_classifier_state = extract_classifier_state(self.model)
                self.model = None
                return {
                    'lora_A_state': local_lora_A_state,
                    'lora_B_state': local_lora_B_state,
                    'classifier_state': local_classifier_state,
                    'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
                }
            if getattr(self.args, 'algo', 'ferret') == 'florg':
                local_florg_A_state = extract_florg_A_state(self.model)
                self._release_florg_training_state(framework)
                return {
                    'florg_A': local_florg_A_state,
                    'num_samples': len(self.train_loader),
                    'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
                }

            local_lora_state = extract_lora_state(self.model)
            self.model = None
            return {
                'lora_state': local_lora_state,
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        if getattr(self.args, 'algo', 'ferret') == 'fedavg':
            if fedavg_global_state is not None:
                self.model.load_state_dict(fedavg_global_state, strict=True)

            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            self.model.train()
            self.model.zero_grad()

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
            else:
                iter_steps = max(int(self.args.local_step), 1)

            loss_total_train = 0.0
            num_trained = 0
            progress_bar = tqdm(range(iter_steps))

            if self.args.batch_or_epoch == 'epoch':
                for cur_step, batch in enumerate(self.train_loader):
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
            else:
                for cur_step in range(iter_steps):
                    batch = self._next_batch()
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )

            local_state = {}
            for name, tensor in self.model.state_dict().items():
                if isinstance(tensor, torch.Tensor):
                    local_state[name] = tensor.detach().cpu().clone()
            self.model = None
            return {
                'model_state_dict': local_state,
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        old_params = [(name, deepcopy(param.data)) for name, param in self.model.named_parameters() if param.requires_grad]

        # initialize a seed pool
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        lr = self.args.lr

        if self.args.batch_or_epoch == 'epoch':
            iter_steps = len(self.train_loader)
        else:
            iter_steps = self.args.local_step

        # Ferret Framework
        self._set_runtime_debug_context(cur_round)
        framework = FerretFramework(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds)
        self.model.train()
        self.model.zero_grad()
        loss_total_train = 0.0
        num_trained = 0

        if self.args.batch_or_epoch == 'epoch':
            progress_bar = tqdm(range(iter_steps))
            for cur_step, batch in enumerate(self.train_loader):
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }
                if cur_step % self.args.n_accum == self.args.n_accum - 1:
                    apply_optim_step = True
                else:
                    apply_optim_step = False
                _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                progress_bar.update(1)
                if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                    loss_total_train += loss
                    num_trained += len(batch['input_ids'])
                progress_bar.set_description(
                    f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                )
        else:
            progress_bar = tqdm(range(iter_steps))
            for cur_step in range(iter_steps):
                batch = self._next_batch()

                if cur_step % self.args.n_accum == self.args.n_accum - 1:
                    apply_optim_step = True
                else:
                    apply_optim_step = False

                _, loss = framework.step(batch, apply_optim_step=apply_optim_step)

                progress_bar.update(1)
                if torch.isfinite(loss) and (self.args.grad_clip <= 0 or loss != 0.0):
                    loss_total_train += loss
                    num_trained += len(batch['input_ids'])
                progress_bar.set_description(
                    f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                )

        self.local_seed_pool = framework.project_update(dict(old_params))

        # save both CPU and GPU memory
        del old_params, framework
        self.model = None
        return float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0

    def clear_model(self):
        # clear model to save memory
        self.model = None

    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model
