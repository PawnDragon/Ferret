import random
import math
import numpy as np
import torch
import torch.nn.functional as F

from optimizers.submuon_utils import make_uv, zeropower_via_newtonschulz5, select_target_linear_layers


def _to_python_int_step(step_value):
    if isinstance(step_value, torch.Tensor):
        if step_value.numel() == 0:
            return 0
        return int(step_value.item())
    try:
        return int(step_value)
    except (TypeError, ValueError):
        return 0


def _is_finite_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all().item())


def export_named_adamw_state(model, optimizer):
    named_state = {'state': {}}
    if optimizer is None:
        return named_state

    id_to_name = {id(param): name for name, param in model.named_parameters()}
    for param_obj, state in optimizer.state.items():
        param_name = id_to_name.get(id(param_obj), None)
        if param_name is None or (not isinstance(state, dict)):
            continue

        exp_avg = state.get('exp_avg', None)
        exp_avg_sq = state.get('exp_avg_sq', None)
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            continue
        if (not _is_finite_tensor(exp_avg)) or (not _is_finite_tensor(exp_avg_sq)):
            continue

        state_entry = {
            'step': _to_python_int_step(state.get('step', 0)),
            'exp_avg': exp_avg.detach().cpu().clone(),
            'exp_avg_sq': exp_avg_sq.detach().cpu().clone(),
        }
        max_exp_avg_sq = state.get('max_exp_avg_sq', None)
        if isinstance(max_exp_avg_sq, torch.Tensor):
            state_entry['max_exp_avg_sq'] = max_exp_avg_sq.detach().cpu().clone()

        named_state['state'][param_name] = state_entry

    return named_state


def load_named_adamw_state(model, optimizer, named_state):
    if optimizer is None or named_state is None:
        return
    if not isinstance(named_state, dict):
        return
    state_dict = named_state.get('state', {})
    if not isinstance(state_dict, dict):
        return

    name_to_param = dict(model.named_parameters())
    for param_name, state_entry in state_dict.items():
        if not isinstance(state_entry, dict):
            continue

        param_obj = name_to_param.get(param_name, None)
        if param_obj is None:
            continue

        exp_avg = state_entry.get('exp_avg', None)
        exp_avg_sq = state_entry.get('exp_avg_sq', None)
        if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
            continue
        if (not _is_finite_tensor(exp_avg)) or (not _is_finite_tensor(exp_avg_sq)):
            continue

        opt_state = optimizer.state[param_obj]
        opt_state['exp_avg'] = exp_avg.detach().to(device=param_obj.device).clone()
        opt_state['exp_avg_sq'] = exp_avg_sq.detach().to(device=param_obj.device).clone()
        step_int = _to_python_int_step(state_entry.get('step', 0))
        old_step = opt_state.get('step', None)
        if isinstance(old_step, torch.Tensor):
            opt_state['step'] = torch.tensor(step_int, device=old_step.device, dtype=old_step.dtype)
        else:
            opt_state['step'] = step_int

        max_exp_avg_sq = state_entry.get('max_exp_avg_sq', None)
        if isinstance(max_exp_avg_sq, torch.Tensor):
            opt_state['max_exp_avg_sq'] = max_exp_avg_sq.detach().to(device=param_obj.device).clone()


class FerretFramework(object):
    def __init__(self, model, args, lr, candidate_seeds):
        self.args = args
        self.lr = lr
        self.model = model
        self.algo = getattr(args, 'algo', 'ferret')
        self.named_parameters_to_optim = [
            (name, param) for name, param in self.model.named_parameters() if param.requires_grad
        ]
        self.candidate_seeds = candidate_seeds

        self.param_groups = []
        self.optim = None

        # FedSubMuon runtime state
        self.submuon_x = {}
        self.submuon_m = {}
        self.submuon_v = {}
        self.submuon_seeds = {}
        self.subadam_step = 0
        self.target_linear_layers = []
        self._submuon_uv_cache = {}
        self._orig_linear_forward = {}
        self._flora_delta = {}
        self._flora_scaling = 1.0
        self._orig_flora_forward = {}
        self.runtime_client_idx = int(getattr(args, '_runtime_client_idx', -1))
        self.runtime_round_idx = int(getattr(args, '_runtime_round_idx', -1))
        self.debug_nan = bool(getattr(args, 'debug_nan', False))
        self.debug_nan_first_steps = max(int(getattr(args, 'debug_nan_first_steps', 1)), 0)
        self.debug_nan_client_idx = int(getattr(args, 'debug_nan_client_idx', 0))
        self.debug_nan_skip_optim = bool(getattr(args, 'debug_nan_skip_optim', False))
        self._local_step_counter = 0

        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            self._freeze_backbone_for_submuon()
            self.target_linear_layers = select_target_linear_layers(self.model, self.args.rank_r)
        elif self.algo == 'fedavg':
            self.optim = torch.optim.AdamW(
                [p for _, p in self.named_parameters_to_optim],
                lr=args.lr,
                betas=(
                    float(getattr(args, 'adam_beta1', 0.9)),
                    float(getattr(args, 'adam_beta2', 0.999)),
                ),
                eps=float(getattr(args, 'adam_eps', 1e-8)),
                weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            )
        elif self.algo in ['fedit', 'flora']:
            self.optim = torch.optim.AdamW(
                [p for _, p in self.named_parameters_to_optim],
                lr=args.lr,
                betas=(
                    float(getattr(args, 'adam_beta1', 0.9)),
                    float(getattr(args, 'adam_beta2', 0.999)),
                ),
                eps=float(getattr(args, 'adam_eps', 1e-8)),
                weight_decay=float(getattr(args, 'weight_decay', 0.0)),
            )
        else:
            self.optim = torch.optim.SGD(
                [p for _, p in self.named_parameters_to_optim],
                lr=args.lr,
                momentum=0.0,
                weight_decay=args.weight_decay,
            )
            self.param_groups = self._group_parameters()

    def _freeze_backbone_for_submuon(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _group_parameters(self):
        # Group parameters with similar dimensions
        groups = []
        current_group = []
        current_dim = 0
        target_dim = int(0.9 * np.median([p.numel() for _, p in self.named_parameters_to_optim]))
        for name, param in self.named_parameters_to_optim:
            param_dim = param.numel()
            if current_dim + param_dim > target_dim and current_group:
                groups.append(current_group)
                current_group = []
                current_dim = 0
            current_group.append((name, param))
            current_dim += param_dim

        if current_group:
            groups.append(current_group)

        return groups

    def _resolve_module(self, module_name):
        module = self.model
        for part in module_name.split('.'):
            module = getattr(module, part)
        return module

    def _get_uv(self, layer_name, out_dim, in_dim, device, dtype):
        key = (layer_name, self.submuon_seeds[layer_name], out_dim, in_dim, self.args.rank_r, str(device), str(dtype))
        if key not in self._submuon_uv_cache:
            self._submuon_uv_cache[key] = make_uv(
                seed=self.submuon_seeds[layer_name],
                out_dim=out_dim,
                in_dim=in_dim,
                r=self.args.rank_r,
                device=device,
                dtype=dtype,
            )
        return self._submuon_uv_cache[key]

    def _install_submuon_forward(self):
        for layer_name in self.target_linear_layers:
            if layer_name not in self.submuon_x:
                continue
            module = self._resolve_module(layer_name)
            if layer_name in self._orig_linear_forward:
                continue
            self._orig_linear_forward[layer_name] = module.forward

            def patched_forward(x, _module=module, _layer_name=layer_name):
                input_shape = x.shape
                x2 = x.reshape(-1, input_shape[-1])
                y0 = F.linear(x2, _module.weight, _module.bias)

                X = self.submuon_x[_layer_name].to(dtype=x2.dtype)
                U, V = self._get_uv(
                    _layer_name,
                    _module.out_features,
                    _module.in_features,
                    x2.device,
                    x2.dtype,
                )
                a = x2 @ V
                b = a @ X.t()
                y1 = b @ U.t()
                y = y0 + y1
                return y.reshape(*input_shape[:-1], _module.out_features)

            module.forward = patched_forward

    def clear_submuon_state(self):
        for layer_name, orig_forward in self._orig_linear_forward.items():
            module = self._resolve_module(layer_name)
            module.forward = orig_forward
        self._orig_linear_forward = {}
        self._submuon_uv_cache = {}
        self.submuon_x = {}
        self.submuon_m = {}
        self.submuon_v = {}
        self.submuon_seeds = {}
        self.subadam_step = 0

    def clear_flora_delta_state(self):
        with torch.no_grad():
            for layer_name, delta_scaled in self._flora_delta.items():
                try:
                    module = self._resolve_module(layer_name)
                except AttributeError:
                    continue
                if hasattr(module, 'weight') and module.weight.shape == delta_scaled.shape:
                    module.weight.data.sub_(delta_scaled.to(device=module.weight.device, dtype=module.weight.dtype))

        for layer_name, orig_forward in self._orig_flora_forward.items():
            module = self._resolve_module(layer_name)
            module.forward = orig_forward
        self._orig_flora_forward = {}
        self._flora_delta = {}
        self._flora_scaling = 1.0

    def set_flora_delta_state(self, delta_state, scaling=1.0):
        self.clear_flora_delta_state()
        if delta_state is None:
            return

        self._flora_scaling = float(scaling)
        with torch.no_grad():
            for layer_name, delta in delta_state.items():
                try:
                    module = self._resolve_module(layer_name)
                except AttributeError:
                    continue
                if not hasattr(module, 'weight'):
                    continue

                delta_scaled = (delta.detach().to(
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                ) * self._flora_scaling).contiguous()

                if module.weight.shape != delta_scaled.shape:
                    continue
                module.weight.data.add_(delta_scaled)
                self._flora_delta[layer_name] = delta_scaled

    def set_submuon_state(self, x_state, m_state, seeds, trainable=True, v_state=None):
        if self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            return
        self.clear_submuon_state()

        device = next(self.model.parameters()).device
        self.submuon_x = {}
        self.submuon_m = {}
        self.submuon_v = {}
        self.submuon_seeds = dict(seeds)

        for layer_name in self.target_linear_layers:
            if layer_name not in x_state:
                continue
            x_tensor = x_state[layer_name].to(device=device, dtype=torch.float32)
            self.submuon_x[layer_name] = torch.nn.Parameter(x_tensor.clone().detach(), requires_grad=trainable)
            if m_state is not None and layer_name in m_state:
                self.submuon_m[layer_name] = m_state[layer_name].to(device=device, dtype=torch.float32).clone().detach()
            else:
                self.submuon_m[layer_name] = torch.zeros_like(self.submuon_x[layer_name], device=device, dtype=torch.float32)
            if v_state is not None and layer_name in v_state:
                self.submuon_v[layer_name] = v_state[layer_name].to(device=device, dtype=torch.float32).clone().detach()
            else:
                self.submuon_v[layer_name] = torch.zeros_like(self.submuon_m[layer_name])

        self._install_submuon_forward()

    def export_submuon_state(self, with_v=False, with_m=True):
        x_out = {k: v.detach().cpu().clone() for k, v in self.submuon_x.items()}
        if not with_m and not with_v:
            return x_out
        m_out = {k: v.detach().cpu().clone() for k, v in self.submuon_m.items()}
        if with_v:
            v_out = {k: v.detach().cpu().clone() for k, v in self.submuon_v.items()}
            return x_out, m_out, v_out
        return x_out, m_out

    def get_named_optim_state(self):
        return export_named_adamw_state(self.model, self.optim)

    def load_named_optim_state(self, named_state):
        load_named_adamw_state(self.model, self.optim, named_state)

    def _debug_active_for_this_client(self):
        if not self.debug_nan:
            return False
        if self.debug_nan_client_idx < 0:
            return True
        return self.runtime_client_idx == self.debug_nan_client_idx

    def _should_trace_this_step(self):
        if not self._debug_active_for_this_client():
            return False
        return self._local_step_counter <= self.debug_nan_first_steps

    def _safe_tensor_stats(self, tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            return {'shape': None, 'finite_ratio': float('nan'), 'abs_max': float('nan'), 'mean': float('nan')}
        data = tensor.detach()
        finite_mask = torch.isfinite(data)
        finite_ratio = float(finite_mask.float().mean().item())
        finite_vals = data[finite_mask]
        if finite_vals.numel() == 0:
            return {'shape': tuple(data.shape), 'finite_ratio': finite_ratio, 'abs_max': float('nan'), 'mean': float('nan')}
        finite_vals = finite_vals.float()
        return {
            'shape': tuple(data.shape),
            'finite_ratio': finite_ratio,
            'abs_max': float(finite_vals.abs().max().item()),
            'mean': float(finite_vals.mean().item()),
        }

    def _batch_debug_summary(self, batch):
        labels = batch.get('labels', None)
        input_ids = batch.get('input_ids', None)
        attention_mask = batch.get('attention_mask', None)
        vocab_size = int(getattr(getattr(self.model, 'config', None), 'vocab_size', -1))

        valid_label_count = -1
        valid_label_min = None
        valid_label_max = None
        if isinstance(labels, torch.Tensor):
            valid_mask = labels.ne(-100)
            valid_label_count = int(valid_mask.sum().item())
            if valid_label_count > 0:
                valid_vals = labels[valid_mask]
                valid_label_min = int(valid_vals.min().item())
                valid_label_max = int(valid_vals.max().item())

        attention_tokens = int(attention_mask.sum().item()) if isinstance(attention_mask, torch.Tensor) else -1
        input_min = int(input_ids.min().item()) if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0 else None
        input_max = int(input_ids.max().item()) if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0 else None
        oov_count = 0
        if isinstance(input_ids, torch.Tensor) and input_ids.numel() > 0 and vocab_size > 0:
            oov_mask = (input_ids < 0) | (input_ids >= vocab_size)
            oov_count = int(oov_mask.sum().item())

        return {
            'input_shape': tuple(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None,
            'labels_shape': tuple(labels.shape) if isinstance(labels, torch.Tensor) else None,
            'attention_shape': tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None,
            'attention_tokens': attention_tokens,
            'input_min': input_min,
            'input_max': input_max,
            'valid_label_count': valid_label_count,
            'valid_label_min': valid_label_min,
            'valid_label_max': valid_label_max,
            'vocab_size': vocab_size,
            'oov_count': oov_count,
        }

    def _grad_debug_summary(self):
        total_grad_params = 0
        nonfinite_grad_names = []
        total_sq = 0.0
        max_abs = 0.0

        for name, param in self.named_parameters_to_optim:
            grad = param.grad
            if grad is None:
                continue
            total_grad_params += 1
            grad_data = grad.detach()
            finite_mask = torch.isfinite(grad_data)
            if not bool(finite_mask.all().item()):
                if len(nonfinite_grad_names) < 8:
                    nonfinite_grad_names.append(name)
            finite_vals = grad_data[finite_mask]
            if finite_vals.numel() > 0:
                finite_vals = finite_vals.float()
                total_sq += float((finite_vals * finite_vals).sum().item())
                max_abs = max(max_abs, float(finite_vals.abs().max().item()))

        grad_global_norm = math.sqrt(max(total_sq, 0.0))
        return {
            'total_grad_params': int(total_grad_params),
            'nonfinite_grad_count': int(len(nonfinite_grad_names)),
            'nonfinite_grad_names': nonfinite_grad_names,
            'grad_global_norm': float(grad_global_norm),
            'grad_abs_max': float(max_abs),
        }

    def _print_forward_debug(self, batch, logits, loss, stage):
        batch_summary = self._batch_debug_summary(batch)
        logits_stats = self._safe_tensor_stats(logits)
        loss_val = float(loss.detach().item()) if isinstance(loss, torch.Tensor) and loss.numel() == 1 else float('nan')
        print(
            f'[nan-debug][{self.algo}] stage={stage} client={self.runtime_client_idx} round={self.runtime_round_idx} '
            f'local_step={self._local_step_counter} loss={loss_val:.6e} '
            f'logits_finite_ratio={logits_stats["finite_ratio"]:.6f} logits_abs_max={logits_stats["abs_max"]:.6e} '
            f'valid_labels={batch_summary["valid_label_count"]} attention_tokens={batch_summary["attention_tokens"]} '
            f'input_id_range=({batch_summary["input_min"]},{batch_summary["input_max"]}) '
            f'oov_tokens={batch_summary["oov_count"]} vocab_size={batch_summary["vocab_size"]}'
        )
        if batch_summary['valid_label_count'] == 0:
            print('[nan-debug] warning: valid_labels=0 (all labels are ignore_index=-100), CE loss can become NaN.')

    def _print_grad_debug(self, grad_summary, stage):
        print(
            f'[nan-debug][{self.algo}] stage={stage} client={self.runtime_client_idx} round={self.runtime_round_idx} '
            f'local_step={self._local_step_counter} grad_params={grad_summary["total_grad_params"]} '
            f'grad_global_norm={grad_summary["grad_global_norm"]:.6e} grad_abs_max={grad_summary["grad_abs_max"]:.6e} '
            f'nonfinite_grad_count={grad_summary["nonfinite_grad_count"]}'
        )
        if grad_summary['nonfinite_grad_count'] > 0:
            print(f'[nan-debug] nonfinite_grad_names(head): {grad_summary["nonfinite_grad_names"]}')

    def step(self, batch, apply_optim_step=False):
        """
        Perform a training step.
        """
        self._local_step_counter += 1
        logits, loss = self.forward(batch)
        if self._should_trace_this_step():
            self._print_forward_debug(batch, logits, loss, stage='forward_trace')
        if (not torch.isfinite(loss).all().item()) and self._debug_active_for_this_client():
            self._print_forward_debug(batch, logits, loss, stage='nonfinite_loss')
            if self.debug_nan_skip_optim:
                if self.optim is not None:
                    self.optim.zero_grad()
                return logits.detach(), loss.detach()

        if self.algo == 'fedsubmuon':
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                beta = self.args.beta
                with torch.no_grad():
                    for layer_name, X in self.submuon_x.items():
                        grad = X.grad
                        if grad is None:
                            continue
                        self.submuon_m[layer_name].mul_(beta).add_((1.0 - beta) * grad)
                        delta = zeropower_via_newtonschulz5(self.submuon_m[layer_name].float(), self.args.ns_steps)
                        X.data.sub_(self.lr * delta.to(dtype=X.data.dtype))
                        X.grad = None
            return logits.detach(), loss.detach()

        if self.algo == 'fedsubadam':
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                self.subadam_step += 1
                beta1 = self.args.beta1
                beta2 = self.args.beta2
                eps = self.args.eps
                with torch.no_grad():
                    for layer_name, X in self.submuon_x.items():
                        grad = X.grad
                        if grad is None:
                            continue
                        self.submuon_m[layer_name].mul_(beta1).add_((1.0 - beta1) * grad)
                        self.submuon_v[layer_name].mul_(beta2).add_((1.0 - beta2) * (grad * grad))
                        m_hat = self.submuon_m[layer_name] / (1.0 - beta1 ** self.subadam_step)
                        v_hat = self.submuon_v[layer_name] / (1.0 - beta2 ** self.subadam_step)
                        X.data.sub_(self.lr * (m_hat / (torch.sqrt(v_hat) + eps)).to(dtype=X.data.dtype))
                        X.grad = None
            return logits.detach(), loss.detach()

        if self.algo == 'fedsubsgd':
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                with torch.no_grad():
                    for _, X in self.submuon_x.items():
                        grad = X.grad
                        if grad is None:
                            continue
                        X.data.sub_(self.lr * grad.to(dtype=X.data.dtype))
                        X.grad = None
            return logits.detach(), loss.detach()

        if self.algo == 'fedavg':
            (loss / self.args.n_accum).backward()
            grad_summary = self._grad_debug_summary() if self._debug_active_for_this_client() else None
            if grad_summary is not None and self._should_trace_this_step():
                self._print_grad_debug(grad_summary, stage='grad_trace')
            if grad_summary is not None and grad_summary['nonfinite_grad_count'] > 0 and (not self._should_trace_this_step()):
                self._print_grad_debug(grad_summary, stage='nonfinite_grad')
            if grad_summary is not None and grad_summary['nonfinite_grad_count'] > 0 and self.debug_nan_skip_optim:
                if self.optim is not None:
                    self.optim.zero_grad()
                return logits.detach(), loss.detach()

            max_grad_norm = float(getattr(self.args, 'max_grad_norm', -1.0))
            if max_grad_norm <= 0.0 and self.args.grad_clip > 0.0:
                max_grad_norm = float(self.args.grad_clip)
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            if apply_optim_step:
                self.optim.step()
                self.optim.zero_grad()
            return logits.detach(), loss.detach()

        (loss / self.args.n_accum).backward()
        grad_summary = self._grad_debug_summary() if self._debug_active_for_this_client() else None
        if grad_summary is not None and self._should_trace_this_step():
            self._print_grad_debug(grad_summary, stage='grad_trace')
        if grad_summary is not None and grad_summary['nonfinite_grad_count'] > 0 and (not self._should_trace_this_step()):
            self._print_grad_debug(grad_summary, stage='nonfinite_grad')
        if grad_summary is not None and grad_summary['nonfinite_grad_count'] > 0 and self.debug_nan_skip_optim:
            if self.optim is not None:
                self.optim.zero_grad()
            return logits.detach(), loss.detach()

        if self.args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

        if apply_optim_step:
            self.optim.step()
            self.optim.zero_grad()

        return logits.detach(), loss.detach()

    def project_update(self, old_params):
        with torch.no_grad():
            coordinate_of_seeds = []
            total_coordinates = 0

            max_delta_norm = 1e-8
            for group in self.param_groups:
                group_delta = torch.cat([old_params[name].to(param.data.device).flatten() - param.data.flatten() for name, param in group])
                delta_norm = group_delta.norm()
                if (not torch.isnan(delta_norm).any().item()) and (delta_norm.item() > max_delta_norm):
                    max_delta_norm = delta_norm.item()

            for group in self.param_groups:
                group_delta = torch.cat([old_params[name].to(param.data.device).flatten() - param.data.flatten() for name, param in group])
                group_size = group_delta.numel()

                coordinate = torch.zeros(self.args.K, device=group_delta.device, dtype=group_delta.dtype)
                if torch.isnan(group_delta.norm()).any().item():
                    max_n_seeds = 2
                else:
                    max_n_seeds = max(int(group_delta.norm().item() / max_delta_norm * self.args.K), 2)

                total_coordinates += max_n_seeds

                seed_idxs = random.sample(range(self.args.K), max_n_seeds)
                for idx in seed_idxs:
                    seed = self.candidate_seeds[idx]
                    sqrt_d = 1 / group_size ** 0.5
                    torch.manual_seed(seed)
                    torch.cuda.random.manual_seed(seed)
                    base = torch.empty(group_size, device=group_delta.device, dtype=group_delta.dtype)
                    base = torch.nn.init.trunc_normal_(base, mean=0.0, std=1.0, a=-sqrt_d, b=sqrt_d)
                    coordinate[idx] = torch.sum(group_delta * base)

                coordinate *= group_size / max_n_seeds
                coordinate_of_seeds.append(coordinate)

            print('total coordinates to send:', total_coordinates)
            self.local_seed_pool = {}
            coordinate_of_seeds = torch.stack(coordinate_of_seeds, dim=1)
            for i, seed in enumerate(self.candidate_seeds):
                self.local_seed_pool[seed] = coordinate_of_seeds[i]
        return self.local_seed_pool

    def forward(self, batch):
        """
        Forward pass to compute the loss.
        """
        outputs = self.model(**batch)
        logits = outputs.logits
        loss = outputs.loss
        return logits, loss

    def update(self, seed=None, grad=None, max_norm=10):
        """
        Update the parameters using the true/estimated gradients.
        """
        with torch.no_grad():
            grad_idx = 0
            for group in self.param_groups:
                group_size = sum(param.numel() for _, param in group)
                sqrt_d = group_size ** -0.5
                torch.manual_seed(seed)
                torch.cuda.random.manual_seed(seed)
                base = torch.empty(group_size, device=grad[grad_idx].device, dtype=grad[grad_idx].dtype)
                base = torch.nn.init.trunc_normal_(base, mean=0.0, std=1.0, a=-sqrt_d, b=sqrt_d)
                base.mul_(grad[grad_idx])
                if torch.isfinite(base).all():
                    total_norm = torch.linalg.norm(base)
                    if torch.isfinite(total_norm):
                        clip_coef = max_norm / (total_norm + 1e-8)
                        if clip_coef < 1:
                            base.mul_(clip_coef)
                        start = 0
                        for _, param in group:
                            end = start + param.numel()
                            param_update = base[start:end].reshape(param.shape)
                            param.data.sub_(self.args.slr * param_update)
                            start = end
                grad_idx += 1
