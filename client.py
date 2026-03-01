from tqdm import tqdm
from copy import deepcopy

from optimizers.ferret_optimizer import *
from optimizers.lora_utils import (
    build_lora_model,
    extract_lora_A_state,
    extract_lora_B_state,
    extract_lora_state,
    load_lora_A_state,
    load_lora_B_state,
    load_lora_state,
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
        self.local_lora_B_state = None
        self.prev_round_lora_A_state = None

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

    def local_train_with_seed_pool(
        self,
        pulled_model,
        cur_round,
        submuon_state=None,
        lora_state=None,
        lora_A_state=None,
        fedavg_global_state=None,
        global_named_optim_state=None,
    ):
        self.model = pulled_model
        self.model.to(self.device)

        if getattr(self.args, 'algo', 'ferret') in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                x_state=submuon_state['x_global'],
                m_state=submuon_state.get('m_global', None),
                seeds=submuon_state['seeds'],
                trainable=True,
                v_state=submuon_state.get('v_global', None),
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

            if getattr(self.args, 'algo', 'ferret') == 'fedsubadam':
                x_local, m_local, v_local = framework.export_submuon_state(with_v=True)
            elif getattr(self.args, 'algo', 'ferret') == 'fedsubsgd':
                x_local = framework.export_submuon_state(with_m=False)
            else:
                x_local, m_local = framework.export_submuon_state()
            framework.clear_submuon_state()
            self.model = None
            payload = {'x': x_local, 'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0}
            if getattr(self.args, 'algo', 'ferret') == 'fedsubadam':
                payload['m'] = m_local
                payload['v'] = v_local
            elif getattr(self.args, 'algo', 'ferret') == 'fedsubmuon':
                payload['m'] = m_local
            return payload

        if getattr(self.args, 'algo', 'ferret') in ['fedit', 'flora', 'fedsalora']:
            self.model = build_lora_model(self.model, self.args)
            self.model.to(self.device)
            if getattr(self.args, 'algo', 'ferret') == 'fedit':
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

            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            if getattr(self.args, 'algo', 'ferret') == 'fedit' and global_named_optim_state is not None:
                framework.load_named_optim_state(global_named_optim_state)
                self._maybe_log_loaded_named_state(framework, global_named_optim_state, cur_round, tag='fedit')
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

            local_lora_state = extract_lora_state(self.model)
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

            named_optim_state = framework.get_named_optim_state() if getattr(self.args, 'algo', 'ferret') == 'fedit' else None
            self.model = None
            return {
                'lora_state': local_lora_state,
                'named_optim_state': named_optim_state,  # FLoRA keeps per-round re-init, no aligned optimizer state upload.
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        if getattr(self.args, 'algo', 'ferret') == 'fedavg':
            if fedavg_global_state is not None:
                self.model.load_state_dict(fedavg_global_state, strict=True)

            self._set_runtime_debug_context(cur_round)
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            if global_named_optim_state is not None:
                framework.load_named_optim_state(global_named_optim_state)
                self._maybe_log_loaded_named_state(framework, global_named_optim_state, cur_round, tag='fedavg')
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
            named_optim_state = framework.get_named_optim_state()
            self.model = None
            return {
                'model_state_dict': local_state,
                'named_optim_state': named_optim_state,
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
