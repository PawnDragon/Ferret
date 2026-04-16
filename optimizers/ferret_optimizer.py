import random
import math
import numpy as np
import torch
import torch.nn.functional as F

from optimizers.fedmultisub_utils import (
    build_adamss_allocator,
    compute_adamss_subspace_score,
    shape_signature as multisub_shape_signature,
)
from optimizers.fedstruct_utils import shape_signature as struct_shape_signature
from optimizers.fedkrso_utils import make_krso_projection
from optimizers.muon_utils import build_muon_optimizer
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


def _is_classifier_param_name(name):
    lowered = str(name).lower()
    return ('classifier' in lowered) or lowered.startswith('score.') or ('.score.' in lowered)


def resolve_submuon_optimizer_name(args, algo=None):
    algo_name = str(algo if algo is not None else getattr(args, 'algo', '')).lower()
    if algo_name not in ['fedsubmuon', 'fedsubmuon_gt']:
        return None
    name = str(getattr(args, 'optimizer', 'muon')).lower()
    if name not in ['muon', 'adamw', 'sgd']:
        return 'muon'
    return name


def should_aggregate_submuon_m_state(args, algo=None):
    algo_name = str(algo if algo is not None else getattr(args, 'algo', '')).lower()
    return bool(
        algo_name == 'fedsubmuon'
        and resolve_submuon_optimizer_name(args, algo_name) == 'muon'
        and getattr(args, 'aggregate_muon_state', False)
    )


def _build_adamw_step_tensor(optimizer, param_obj, step_int, old_step=None):
    if isinstance(old_step, torch.Tensor):
        return torch.tensor(float(step_int), device=old_step.device, dtype=old_step.dtype)

    step_device = torch.device('cpu')
    for group in optimizer.param_groups:
        params = group.get('params', [])
        if any(p is param_obj for p in params):
            capturable = bool(group.get('capturable', False))
            fused = bool(group.get('fused', False))
            if capturable or fused:
                step_device = param_obj.device
            break
    return torch.tensor(float(step_int), device=step_device, dtype=torch.float32)


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
        # Keep optimizer state dtype aligned with parameter/grad dtype so AdamW foreach path stays valid.
        target_dtype = param_obj.dtype
        opt_state['exp_avg'] = exp_avg.detach().to(device=param_obj.device, dtype=target_dtype).clone()
        opt_state['exp_avg_sq'] = exp_avg_sq.detach().to(device=param_obj.device, dtype=target_dtype).clone()
        step_int = _to_python_int_step(state_entry.get('step', 0))
        old_step = opt_state.get('step', None)
        opt_state['step'] = _build_adamw_step_tensor(
            optimizer=optimizer,
            param_obj=param_obj,
            step_int=step_int,
            old_step=old_step,
        )

        max_exp_avg_sq = state_entry.get('max_exp_avg_sq', None)
        if isinstance(max_exp_avg_sq, torch.Tensor):
            opt_state['max_exp_avg_sq'] = max_exp_avg_sq.detach().to(
                device=param_obj.device,
                dtype=target_dtype,
            ).clone()


class FerretFramework(object):
    def __init__(self, model, args, lr, candidate_seeds):
        self.args = args
        self.lr = lr
        self.model = model
        self.algo = getattr(args, 'algo', 'ferret')
        if self.algo in ['fedsalora', 'fedexlora', 'florg']:
            for name, param in self.model.named_parameters():
                if self.algo == 'florg':
                    keep_trainable = 'florg_A' in name
                else:
                    keep_trainable = 'lora_' in name
                if self.algo in ['fedexlora', 'florg'] and _is_classifier_param_name(name):
                    keep_trainable = True
                if not keep_trainable:
                    param.requires_grad_(False)
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
        self.submuon_u_basis = {}
        self.submuon_v_basis = {}
        self.subadam_step = 0
        self.target_linear_layers = []
        self._submuon_uv_cache = {}
        self.multisub_b = {}
        self.multisub_c = {}
        self.multisub_mb = {}
        self.multisub_mc = {}
        self.multisub_meta = {}
        self.multisub_scores = {}
        self.multisub_layer_to_keys = {}
        self._orig_multisub_forward = {}
        self._multisub_allocator = None
        self._multisub_debug_logged = False
        self.struct_x = {}
        self.struct_m = {}
        self.struct_meta = {}
        self.struct_scores = {}
        self.struct_exp_avg_ipt = {}
        self.struct_exp_avg_unc = {}
        self.struct_layer_to_keys = {}
        self._orig_struct_forward = {}
        self._struct_debug_logged = False
        self.krso_seed_pool = []
        self.krso_seed_set = set()
        self.krso_local_b = {}
        self.krso_seed_usage = {}
        self.krso_interval_seed = None
        self.krso_interval_b = {}
        self.krso_interval_p = {}
        self.krso_interval_m1 = {}
        self.krso_interval_m2 = {}
        self.krso_interval_step = {}
        self.krso_num_intervals = 0
        self._orig_krso_forward = {}
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

        if self.algo in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd', 'fedkrso']:
            self._freeze_backbone_for_submuon()
            self.target_linear_layers = select_target_linear_layers(
                self.model,
                self.args.rank_r,
                raw_target_modules=getattr(self.args, 'lora_target_modules', None),
            )
            if len(self.target_linear_layers) == 0:
                raise RuntimeError(
                    f'[{self.algo}] no target linear layer is selected; '
                    f'rank_r={self.args.rank_r}, '
                    f'lora_target_modules={getattr(self.args, "lora_target_modules", None)}'
                )
        elif self.algo in ['fedmultisubmuon', 'fedstructmuon']:
            self._freeze_backbone_for_submuon()
        else:
            self.optim = self._build_local_optimizer([p for _, p in self.named_parameters_to_optim])
            # Ferret still needs grouped params for random-seed projection.
            self.param_groups = self._group_parameters()

    def _freeze_backbone_for_submuon(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _resolve_optimizer_name(self):
        name = str(getattr(self.args, 'optimizer', 'adamw')).lower()
        if name in ['adamw', 'sgd']:
            return name
        if name == 'muon' and self.algo == 'fedit':
            return 'muon'
        return 'adamw'

    def _resolve_submuon_optimizer_name(self):
        return resolve_submuon_optimizer_name(self.args, self.algo)

    def _build_local_optimizer(self, params):
        params = list(params)
        if len(params) == 0:
            return None
        optimizer_name = self._resolve_optimizer_name()
        if optimizer_name == 'muon':
            return build_muon_optimizer(
                params,
                lr=self.args.lr,
                momentum=float(getattr(self.args, 'momentum', 0.95)),
                weight_decay=float(getattr(self.args, 'weight_decay', 0.0)),
                ns_steps=int(getattr(self.args, 'ns_steps', 5)),
            )
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(
                params,
                lr=self.args.lr,
                betas=(
                    float(getattr(self.args, 'adam_beta1', 0.9)),
                    float(getattr(self.args, 'adam_beta2', 0.999)),
                ),
                eps=float(getattr(self.args, 'adam_eps', 1e-8)),
                weight_decay=float(getattr(self.args, 'weight_decay', 0.0)),
            )
        return torch.optim.SGD(
            params,
            lr=self.args.lr,
            momentum=float(getattr(self.args, 'momentum', 0.0)),
            weight_decay=float(getattr(self.args, 'weight_decay', 0.0)),
        )

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
        if layer_name in self.submuon_u_basis and layer_name in self.submuon_v_basis:
            U = self.submuon_u_basis[layer_name].to(device=device, dtype=dtype)
            V = self.submuon_v_basis[layer_name].to(device=device, dtype=dtype)
            return U, V
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

    def get_submuon_uv(self, layer_name, out_dim, in_dim, device, dtype):
        return self._get_uv(layer_name, out_dim, in_dim, device, dtype)

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

    def _install_krso_forward(self):
        for layer_name in self.target_linear_layers:
            module = self._resolve_module(layer_name)
            if layer_name in self._orig_krso_forward:
                continue
            self._orig_krso_forward[layer_name] = module.forward

            def patched_forward(x, _module=module, _layer_name=layer_name):
                input_shape = x.shape
                x2 = x.reshape(-1, input_shape[-1])
                y0 = F.linear(x2, _module.weight, _module.bias)

                B = self.krso_interval_b.get(_layer_name, None)
                P = self.krso_interval_p.get(_layer_name, None)
                if B is None or P is None:
                    return y0.reshape(*input_shape[:-1], _module.out_features)

                if P.device != x2.device or P.dtype != x2.dtype:
                    P = P.to(device=x2.device, dtype=x2.dtype)
                if B.device != x2.device or B.dtype != x2.dtype:
                    B = B.to(device=x2.device, dtype=x2.dtype)
                a = x2 @ P.t()
                y1 = a @ B.t()
                y = y0 + y1
                return y.reshape(*input_shape[:-1], _module.out_features)

            module.forward = patched_forward

    def _install_multisub_forward(self):
        for layer_name, key_list in self.multisub_layer_to_keys.items():
            if len(key_list) == 0:
                continue
            module = self._resolve_module(layer_name)
            if layer_name in self._orig_multisub_forward:
                continue
            self._orig_multisub_forward[layer_name] = module.forward

            def patched_forward(x, _module=module, _layer_name=layer_name, _key_list=tuple(key_list)):
                input_shape = x.shape
                x2 = x.reshape(-1, input_shape[-1])
                y0 = F.linear(x2, _module.weight, _module.bias)
                y_delta = None

                for sub_key in _key_list:
                    B = self.multisub_b[sub_key].to(dtype=x2.dtype)
                    C = self.multisub_c[sub_key].to(dtype=x2.dtype)
                    meta = self.multisub_meta[sub_key]
                    A = meta['A'].to(dtype=x2.dtype)
                    col_idx = meta['indices']
                    x_sub = torch.index_select(x2, dim=1, index=col_idx)
                    abc_1 = x_sub @ C.t()
                    abc_2 = abc_1 @ B.t()
                    contrib = abc_2 @ A.t()
                    y_delta = contrib if y_delta is None else (y_delta + contrib)

                if y_delta is None:
                    return y0.reshape(*input_shape[:-1], _module.out_features)
                y = y0 + y_delta.to(dtype=y0.dtype)
                return y.reshape(*input_shape[:-1], _module.out_features)

            module.forward = patched_forward

    def _install_struct_forward(self):
        for layer_name, key_list in self.struct_layer_to_keys.items():
            if len(key_list) == 0:
                continue
            module = self._resolve_module(layer_name)
            if layer_name in self._orig_struct_forward:
                continue
            self._orig_struct_forward[layer_name] = module.forward

            def patched_forward(x, _module=module, _layer_name=layer_name, _key_list=tuple(key_list)):
                input_shape = x.shape
                x2 = x.reshape(-1, input_shape[-1])
                y0 = F.linear(x2, _module.weight, _module.bias)
                y_delta = None

                for sub_key in _key_list:
                    X = self.struct_x[sub_key].to(dtype=x2.dtype)
                    meta = self.struct_meta[sub_key]
                    A = meta['A'].to(dtype=x2.dtype)
                    V = meta['V'].to(dtype=x2.dtype)
                    col_idx = meta['indices']
                    x_sub = torch.index_select(x2, dim=1, index=col_idx)
                    ax1 = x_sub @ V
                    ax2 = ax1 @ X.t()
                    contrib = ax2 @ A.t()
                    y_delta = contrib if y_delta is None else (y_delta + contrib)

                if y_delta is None:
                    return y0.reshape(*input_shape[:-1], _module.out_features)
                y = y0 + y_delta.to(dtype=y0.dtype)
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
        self.submuon_u_basis = {}
        self.submuon_v_basis = {}
        self.subadam_step = 0
        if self.algo in ['fedsubmuon', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd']:
            self.optim = None

    def clear_krso_state(self):
        for layer_name, orig_forward in self._orig_krso_forward.items():
            module = self._resolve_module(layer_name)
            module.forward = orig_forward
        self._orig_krso_forward = {}
        self.krso_seed_pool = []
        self.krso_seed_set = set()
        self.krso_local_b = {}
        self.krso_seed_usage = {}
        self.krso_interval_seed = None
        self.krso_interval_b = {}
        self.krso_interval_p = {}
        self.krso_interval_m1 = {}
        self.krso_interval_m2 = {}
        self.krso_interval_step = {}
        self.krso_num_intervals = 0

    def clear_multisub_state(self):
        for layer_name, orig_forward in self._orig_multisub_forward.items():
            module = self._resolve_module(layer_name)
            module.forward = orig_forward
        self._orig_multisub_forward = {}
        self.multisub_b = {}
        self.multisub_c = {}
        self.multisub_mb = {}
        self.multisub_mc = {}
        self.multisub_meta = {}
        self.multisub_scores = {}
        self.multisub_layer_to_keys = {}
        self._multisub_allocator = None

    def clear_struct_state(self):
        for layer_name, orig_forward in self._orig_struct_forward.items():
            module = self._resolve_module(layer_name)
            module.forward = orig_forward
        self._orig_struct_forward = {}
        self.struct_x = {}
        self.struct_m = {}
        self.struct_meta = {}
        self.struct_scores = {}
        self.struct_exp_avg_ipt = {}
        self.struct_exp_avg_unc = {}
        self.struct_layer_to_keys = {}

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

    def set_submuon_state(self, x_state, m_state, seeds, trainable=True, v_state=None, uv_state=None):
        if self.algo not in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd']:
            return
        self.clear_submuon_state()

        device = next(self.model.parameters()).device
        self.submuon_x = {}
        self.submuon_m = {}
        self.submuon_v = {}
        self.submuon_seeds = dict(seeds)
        u_state = {}
        v_basis_state = {}
        if isinstance(uv_state, dict):
            if isinstance(uv_state.get('u', None), dict):
                u_state = uv_state['u']
            if isinstance(uv_state.get('v', None), dict):
                v_basis_state = uv_state['v']

        for layer_name in self.target_linear_layers:
            if layer_name not in x_state:
                continue
            x_tensor = x_state[layer_name].to(device=device, dtype=torch.float32)
            self.submuon_x[layer_name] = torch.nn.Parameter(x_tensor.clone().detach(), requires_grad=trainable)
            if self.algo == 'fedsubmuon' and m_state is not None and layer_name in m_state:
                self.submuon_m[layer_name] = m_state[layer_name].to(device=device, dtype=torch.float32).clone().detach()
            else:
                self.submuon_m[layer_name] = torch.zeros_like(self.submuon_x[layer_name], device=device, dtype=torch.float32)
            if self.algo == 'fedsubmuon' and v_state is not None and layer_name in v_state:
                self.submuon_v[layer_name] = v_state[layer_name].to(device=device, dtype=torch.float32).clone().detach()
            else:
                self.submuon_v[layer_name] = torch.zeros_like(self.submuon_m[layer_name])
            if layer_name in u_state and layer_name in v_basis_state:
                u_tensor = u_state[layer_name]
                v_tensor = v_basis_state[layer_name]
                if isinstance(u_tensor, torch.Tensor) and isinstance(v_tensor, torch.Tensor):
                    self.submuon_u_basis[layer_name] = u_tensor.to(
                        device=device,
                        dtype=torch.float32,
                    ).clone().detach()
                    self.submuon_v_basis[layer_name] = v_tensor.to(
                        device=device,
                        dtype=torch.float32,
                    ).clone().detach()

        self._install_submuon_forward()
        submuon_optimizer = self._resolve_submuon_optimizer_name()
        if self.algo in ['fedsubadam', 'fedsubsgd'] and trainable:
            self.optim = self._build_local_optimizer(self.submuon_x.values())
            if self.optim is not None:
                self.optim.zero_grad()
        elif self.algo in ['fedsubmuon', 'fedsubmuon_gt'] and trainable and submuon_optimizer in ['adamw', 'sgd']:
            self.optim = self._build_local_optimizer(self.submuon_x.values())
            if self.optim is not None:
                self.optim.zero_grad()

    def set_krso_state(self, krso_state, trainable=True):
        if self.algo != 'fedkrso':
            return
        self.clear_krso_state()
        if not isinstance(krso_state, dict):
            return
        seed_pool = krso_state.get('seed_pool', [])
        if not isinstance(seed_pool, (list, tuple)) or len(seed_pool) == 0:
            raise RuntimeError('[fedkrso] krso_state must contain a non-empty seed_pool')
        self.krso_seed_pool = [int(seed) for seed in seed_pool]
        self.krso_seed_set = set(self.krso_seed_pool)
        self.krso_local_b = {}
        self.krso_seed_usage = {int(seed): 0 for seed in self.krso_seed_pool}
        self.krso_interval_seed = None
        self.krso_num_intervals = 0
        if trainable:
            self._install_krso_forward()

    def begin_krso_interval(self, seed):
        if self.algo != 'fedkrso':
            return
        seed = int(seed)
        if seed not in self.krso_seed_set:
            raise RuntimeError(f'[fedkrso] interval seed={seed} not found in current seed pool')
        if self.krso_interval_seed is not None:
            raise RuntimeError('[fedkrso] previous interval is still active when starting a new interval')

        device = next(self.model.parameters()).device
        self.krso_interval_seed = seed
        self.krso_num_intervals += 1
        self.krso_seed_usage[seed] = int(self.krso_seed_usage.get(seed, 0)) + 1
        self.krso_interval_b = {}
        self.krso_interval_p = {}
        self.krso_interval_m1 = {}
        self.krso_interval_m2 = {}
        self.krso_interval_step = {}

        for layer_name in self.target_linear_layers:
            module = self._resolve_module(layer_name)
            out_dim = int(module.out_features)
            in_dim = int(module.in_features)
            rank_eff = int(min(int(self.args.rank_r), out_dim, in_dim))
            if rank_eff <= 0:
                continue
            layer_dtype = module.weight.dtype if module.weight.is_floating_point() else torch.float32
            b_param = torch.nn.Parameter(
                torch.zeros(out_dim, rank_eff, device=device, dtype=layer_dtype),
                requires_grad=True,
            )
            proj = make_krso_projection(
                seed=seed,
                layer_name=layer_name,
                rank=rank_eff,
                in_dim=in_dim,
                device=device,
                dtype=layer_dtype,
            )
            self.krso_interval_b[layer_name] = b_param
            self.krso_interval_p[layer_name] = proj
            self.krso_interval_m1[layer_name] = torch.zeros_like(b_param, device=device, dtype=torch.float32)
            self.krso_interval_m2[layer_name] = torch.zeros_like(b_param, device=device, dtype=torch.float32)
            self.krso_interval_step[layer_name] = 0

    def finish_krso_interval(self):
        if self.algo != 'fedkrso' or self.krso_interval_seed is None:
            return {'seed': None, 'delta_norm': 0.0}

        seed = int(self.krso_interval_seed)
        total_delta_norm = 0.0
        with torch.no_grad():
            for layer_name, b_param in self.krso_interval_b.items():
                proj = self.krso_interval_p.get(layer_name, None)
                if proj is None:
                    continue
                if layer_name not in self.krso_local_b:
                    self.krso_local_b[layer_name] = {}
                if seed not in self.krso_local_b[layer_name]:
                    self.krso_local_b[layer_name][seed] = torch.zeros_like(b_param.detach().cpu(), dtype=torch.float32)
                self.krso_local_b[layer_name][seed].add_(b_param.detach().cpu().to(dtype=torch.float32))

                module = self._resolve_module(layer_name)
                delta = torch.matmul(b_param.detach().float(), proj.detach().float())
                if tuple(delta.shape) != tuple(module.weight.shape):
                    raise RuntimeError(
                        f'[fedkrso] BP shape mismatch @ {layer_name}: '
                        f'got={tuple(delta.shape)}, expected={tuple(module.weight.shape)}'
                    )
                module.weight.data.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
                total_delta_norm += float(torch.linalg.norm(delta).item())

        self.krso_interval_seed = None
        self.krso_interval_b = {}
        self.krso_interval_p = {}
        self.krso_interval_m1 = {}
        self.krso_interval_m2 = {}
        self.krso_interval_step = {}
        return {'seed': seed, 'delta_norm': float(total_delta_norm)}

    def export_krso_state(self):
        out_state = {}
        for layer_name, seed_map in self.krso_local_b.items():
            if not isinstance(seed_map, dict) or len(seed_map) == 0:
                continue
            out_state[layer_name] = {
                int(seed): tensor.detach().cpu().contiguous()
                for seed, tensor in seed_map.items()
                if isinstance(tensor, torch.Tensor)
            }
        used_seeds = sorted(
            [int(seed) for seed, count in self.krso_seed_usage.items() if int(count) > 0]
        )
        return out_state, used_seeds

    def export_submuon_state(self, with_v=False, with_m=True):
        x_out = {k: v.detach().cpu().clone() for k, v in self.submuon_x.items()}
        if not with_m and not with_v:
            return x_out
        m_out = {k: v.detach().cpu().clone() for k, v in self.submuon_m.items()}
        if with_v:
            v_out = {k: v.detach().cpu().clone() for k, v in self.submuon_v.items()}
            return x_out, m_out, v_out
        return x_out, m_out

    def set_multisub_state(self, multisub_state, trainable=True):
        if self.algo != 'fedmultisubmuon':
            return
        self.clear_multisub_state()
        if not isinstance(multisub_state, dict):
            return

        b_state = multisub_state.get('b_global', {})
        c_state = multisub_state.get('c_global', {})
        metadata = multisub_state.get('metadata', {})
        selected_keys = multisub_state.get('selected_keys', [])
        incoming_scores = multisub_state.get('score_state', {})
        if not isinstance(b_state, dict) or not isinstance(c_state, dict) or not isinstance(metadata, dict):
            return
        if not isinstance(selected_keys, (list, tuple)):
            selected_keys = list(b_state.keys())
        if len(selected_keys) == 0:
            selected_keys = list(b_state.keys())

        device = next(self.model.parameters()).device
        for sub_key in selected_keys:
            if sub_key not in b_state or sub_key not in c_state or sub_key not in metadata:
                continue
            meta_entry = metadata[sub_key]
            if not isinstance(meta_entry, dict):
                continue
            layer_name = meta_entry.get('layer_name', None)
            a_tensor = meta_entry.get('A', None)
            col_idx = meta_entry.get('indices', None)
            if layer_name is None or (not isinstance(a_tensor, torch.Tensor)) or (not isinstance(col_idx, torch.Tensor)):
                continue

            b_tensor = b_state[sub_key]
            c_tensor = c_state[sub_key]
            if (not isinstance(b_tensor, torch.Tensor)) or (not isinstance(c_tensor, torch.Tensor)):
                continue
            module = self._resolve_module(str(layer_name))
            if b_tensor.ndim != 2 or c_tensor.ndim != 2 or a_tensor.ndim != 2:
                raise RuntimeError(
                    f'[fedmultisubmuon] expected 2D tensors for {sub_key}, '
                    f'got A={tuple(a_tensor.shape)}, B={tuple(b_tensor.shape)}, C={tuple(c_tensor.shape)}'
                )
            rank_big = int(a_tensor.shape[1])
            rank_small = int(c_tensor.shape[0])
            n_sub = int(col_idx.numel())
            if tuple(b_tensor.shape) != (rank_big, rank_small):
                raise RuntimeError(
                    f'[fedmultisubmuon] B shape mismatch for {sub_key}: '
                    f'expected {(rank_big, rank_small)}, got {tuple(b_tensor.shape)}'
                )
            if tuple(c_tensor.shape) != (rank_small, n_sub):
                raise RuntimeError(
                    f'[fedmultisubmuon] C shape mismatch for {sub_key}: '
                    f'expected {(rank_small, n_sub)}, got {tuple(c_tensor.shape)}'
                )
            if int(a_tensor.shape[0]) != int(module.out_features):
                raise RuntimeError(
                    f'[fedmultisubmuon] A out-dim mismatch for {sub_key}: '
                    f'A.shape[0]={int(a_tensor.shape[0])}, layer_out={int(module.out_features)}'
                )
            if col_idx.ndim != 1:
                raise RuntimeError(
                    f'[fedmultisubmuon] indices must be 1D for {sub_key}, got shape={tuple(col_idx.shape)}'
                )
            if int(col_idx.numel()) == 0:
                continue
            if int(col_idx.max().item()) >= int(module.in_features) or int(col_idx.min().item()) < 0:
                raise RuntimeError(
                    f'[fedmultisubmuon] column indices out of range for {sub_key}: '
                    f'min={int(col_idx.min().item())}, max={int(col_idx.max().item())}, in_features={int(module.in_features)}'
                )
            b_param = torch.nn.Parameter(
                b_tensor.to(device=device, dtype=torch.float32).clone().detach(),
                requires_grad=trainable,
            )
            c_param = torch.nn.Parameter(
                c_tensor.to(device=device, dtype=torch.float32).clone().detach(),
                requires_grad=trainable,
            )
            self.multisub_b[sub_key] = b_param
            self.multisub_c[sub_key] = c_param
            self.multisub_mb[sub_key] = torch.zeros_like(b_param, device=device, dtype=torch.float32)
            self.multisub_mc[sub_key] = torch.zeros_like(c_param, device=device, dtype=torch.float32)
            self.multisub_scores[sub_key] = float(incoming_scores.get(sub_key, 0.0))
            self.multisub_meta[sub_key] = {
                'layer_name': str(layer_name),
                'A': a_tensor.to(device=device, dtype=torch.float32).contiguous(),
                'indices': col_idx.to(device=device, dtype=torch.long).contiguous(),
                'rank_big': rank_big,
                'rank_small': rank_small,
            }
            self.multisub_layer_to_keys.setdefault(str(layer_name), []).append(sub_key)
            if (not self._multisub_debug_logged) and trainable:
                delta = self.multisub_meta[sub_key]['A'] @ b_param @ c_param
                if tuple(delta.shape) != (int(module.out_features), int(col_idx.numel())):
                    raise RuntimeError(
                        f'[fedmultisubmuon] invalid ABC shape for {sub_key}: '
                        f'{multisub_shape_signature(self.multisub_meta[sub_key], b_param, c_param)}'
                    )
                print(
                    f'[debug][fedmultisubmuon][framework] {sub_key} '
                    f'A_shape={tuple(a_tensor.shape)} B_shape={tuple(b_tensor.shape)} C_shape={tuple(c_tensor.shape)}'
                )
                self._multisub_debug_logged = True

        if len(self.multisub_b) > 0 and trainable:
            score_interval = max(int(getattr(self.args, 'multisub_score_interval', 1)), 1)
            beta1 = float(getattr(self.args, 'multisub_score_beta1', 0.9))
            beta2 = float(getattr(self.args, 'multisub_score_beta2', 0.999))
            self._multisub_allocator = build_adamss_allocator(
                mask_interval=score_interval,
                beta1=beta1,
                beta2=beta2,
            )
        if len(self.multisub_b) > 0:
            self._install_multisub_forward()

    def export_multisub_state(self):
        b_out = {k: v.detach().cpu().clone() for k, v in self.multisub_b.items()}
        c_out = {k: v.detach().cpu().clone() for k, v in self.multisub_c.items()}
        score_out = {}
        for sub_key in self.multisub_b.keys():
            if sub_key in self.multisub_scores:
                score_out[sub_key] = float(self.multisub_scores[sub_key])
            else:
                score_out[sub_key] = 0.0
        return b_out, c_out, score_out

    def set_struct_state(self, struct_state, trainable=True):
        if self.algo != 'fedstructmuon':
            return
        self.clear_struct_state()
        if not isinstance(struct_state, dict):
            return

        x_state = struct_state.get('x_global', {})
        metadata = struct_state.get('metadata', {})
        selected_keys = struct_state.get('selected_keys', [])
        incoming_scores = struct_state.get('score_state', {})
        if not isinstance(x_state, dict) or not isinstance(metadata, dict):
            return
        if not isinstance(selected_keys, (list, tuple)):
            selected_keys = list(x_state.keys())
        if len(selected_keys) == 0:
            selected_keys = list(x_state.keys())

        device = next(self.model.parameters()).device
        for sub_key in selected_keys:
            if sub_key not in x_state or sub_key not in metadata:
                continue
            x_tensor = x_state[sub_key]
            meta_entry = metadata[sub_key]
            if (not isinstance(x_tensor, torch.Tensor)) or (not isinstance(meta_entry, dict)):
                continue

            layer_name = meta_entry.get('layer_name', None)
            a_tensor = meta_entry.get('A', None)
            v_tensor = meta_entry.get('V', None)
            col_idx = meta_entry.get('indices', None)
            if (
                layer_name is None
                or (not isinstance(a_tensor, torch.Tensor))
                or (not isinstance(v_tensor, torch.Tensor))
                or (not isinstance(col_idx, torch.Tensor))
            ):
                continue

            module = self._resolve_module(str(layer_name))
            if a_tensor.ndim != 2 or v_tensor.ndim != 2 or x_tensor.ndim != 2:
                raise RuntimeError(
                    f'[fedstructmuon] expected 2D tensors for {sub_key}, '
                    f'got A={tuple(a_tensor.shape)}, X={tuple(x_tensor.shape)}, V={tuple(v_tensor.shape)}'
                )
            if col_idx.ndim != 1:
                raise RuntimeError(
                    f'[fedstructmuon] indices must be 1D for {sub_key}, got shape={tuple(col_idx.shape)}'
                )
            n_sub = int(col_idx.numel())
            if n_sub == 0:
                continue
            rank_left = int(a_tensor.shape[1])
            rank_right = int(v_tensor.shape[1])
            if tuple(x_tensor.shape) != (rank_left, rank_right):
                raise RuntimeError(
                    f'[fedstructmuon] X shape mismatch for {sub_key}: '
                    f'expected {(rank_left, rank_right)}, got {tuple(x_tensor.shape)}'
                )
            if tuple(v_tensor.shape) != (n_sub, rank_right):
                raise RuntimeError(
                    f'[fedstructmuon] V shape mismatch for {sub_key}: '
                    f'expected {(n_sub, rank_right)}, got {tuple(v_tensor.shape)}'
                )
            if int(a_tensor.shape[0]) != int(module.out_features):
                raise RuntimeError(
                    f'[fedstructmuon] A out-dim mismatch for {sub_key}: '
                    f'A.shape[0]={int(a_tensor.shape[0])}, layer_out={int(module.out_features)}'
                )
            if int(col_idx.max().item()) >= int(module.in_features) or int(col_idx.min().item()) < 0:
                raise RuntimeError(
                    f'[fedstructmuon] column indices out of range for {sub_key}: '
                    f'min={int(col_idx.min().item())}, max={int(col_idx.max().item())}, in_features={int(module.in_features)}'
                )

            x_param = torch.nn.Parameter(
                x_tensor.to(device=device, dtype=torch.float32).clone().detach(),
                requires_grad=trainable,
            )
            self.struct_x[sub_key] = x_param
            self.struct_m[sub_key] = torch.zeros_like(x_param, device=device, dtype=torch.float32)
            self.struct_scores[sub_key] = float(incoming_scores.get(sub_key, 0.0))
            self.struct_exp_avg_ipt[sub_key] = torch.zeros_like(x_param, device=device, dtype=torch.float32)
            self.struct_exp_avg_unc[sub_key] = torch.zeros_like(x_param, device=device, dtype=torch.float32)
            self.struct_meta[sub_key] = {
                'layer_name': str(layer_name),
                'A': a_tensor.to(device=device, dtype=torch.float32).contiguous(),
                'V': v_tensor.to(device=device, dtype=torch.float32).contiguous(),
                'indices': col_idx.to(device=device, dtype=torch.long).contiguous(),
                'rank': rank_left,  # backward-compatible alias
                'rank_left': rank_left,
                'rank_right': rank_right,
            }
            self.struct_layer_to_keys.setdefault(str(layer_name), []).append(sub_key)

            if (not self._struct_debug_logged) and trainable:
                delta = self.struct_meta[sub_key]['A'] @ x_param @ self.struct_meta[sub_key]['V'].t()
                if tuple(delta.shape) != (int(module.out_features), int(col_idx.numel())):
                    raise RuntimeError(
                        f'[fedstructmuon] invalid AXV shape for {sub_key}: '
                        f'{struct_shape_signature(self.struct_meta[sub_key], x_param)}'
                    )
                print(
                    f'[debug][fedstructmuon][framework] {sub_key} '
                    f'A_shape={tuple(a_tensor.shape)} X_shape={tuple(x_tensor.shape)} V_shape={tuple(v_tensor.shape)}'
                )
                self._struct_debug_logged = True

        if len(self.struct_x) > 0:
            self._install_struct_forward()

    def export_struct_state(self):
        x_out = {k: v.detach().cpu().clone() for k, v in self.struct_x.items()}
        score_out = {}
        for sub_key in self.struct_x.keys():
            if sub_key in self.struct_scores:
                score_out[sub_key] = float(self.struct_scores[sub_key])
            else:
                score_out[sub_key] = 0.0
        return x_out, score_out

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

        if self.algo == 'fedkrso':
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                beta1 = float(getattr(self.args, 'beta1', 0.9))
                beta2 = float(getattr(self.args, 'beta2', 0.999))
                eps = float(getattr(self.args, 'eps', 1e-8))
                with torch.no_grad():
                    for layer_name, b_param in self.krso_interval_b.items():
                        grad = b_param.grad
                        if grad is None:
                            continue
                        self.krso_interval_step[layer_name] = int(self.krso_interval_step.get(layer_name, 0)) + 1
                        step_idx = int(self.krso_interval_step[layer_name])
                        self.krso_interval_m1[layer_name].mul_(beta1).add_(grad.float(), alpha=(1.0 - beta1))
                        self.krso_interval_m2[layer_name].mul_(beta2).addcmul_(
                            grad.float(),
                            grad.float(),
                            value=(1.0 - beta2),
                        )
                        m_hat = self.krso_interval_m1[layer_name] / (1.0 - (beta1 ** step_idx))
                        v_hat = self.krso_interval_m2[layer_name] / (1.0 - (beta2 ** step_idx))
                        step_update = self.lr * m_hat / (torch.sqrt(v_hat) + eps)
                        updated_b = b_param.data.float().sub_(step_update)
                        b_param.data.copy_(updated_b.to(dtype=b_param.dtype))
                        b_param.grad = None
            return logits.detach(), loss.detach()

        if self.algo in ['fedsubmuon', 'fedsubmuon_gt'] and self._resolve_submuon_optimizer_name() in ['adamw', 'sgd']:
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                if self.optim is not None:
                    self.optim.step()
                    self.optim.zero_grad()
            return logits.detach(), loss.detach()

        if self.algo in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt']:
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

        if self.algo == 'fedmultisubmuon':
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                beta = float(getattr(self.args, 'beta', 0.95))
                score_interval = max(int(getattr(self.args, 'multisub_score_interval', 1)), 1)
                with torch.no_grad():
                    for sub_key in self.multisub_b.keys():
                        B = self.multisub_b[sub_key]
                        C = self.multisub_c[sub_key]
                        grad_b = B.grad
                        grad_c = C.grad
                        if grad_b is None or grad_c is None:
                            continue
                        self.multisub_mb[sub_key].mul_(beta).add_((1.0 - beta) * grad_b)
                        self.multisub_mc[sub_key].mul_(beta).add_((1.0 - beta) * grad_c)
                        delta_b = zeropower_via_newtonschulz5(self.multisub_mb[sub_key].float(), self.args.ns_steps)
                        delta_c = zeropower_via_newtonschulz5(self.multisub_mc[sub_key].float(), self.args.ns_steps)
                        B.data.sub_(self.lr * delta_b.to(dtype=B.data.dtype))
                        C.data.sub_(self.lr * delta_c.to(dtype=C.data.dtype))

                        if (self._local_step_counter % score_interval) == 0:
                            score_val = compute_adamss_subspace_score(
                                allocator=self._multisub_allocator,
                                sub_key=sub_key,
                                b_param=B,
                                c_param=C,
                            )
                            if score_val is not None:
                                self.multisub_scores[sub_key] = float(score_val)
                        B.grad = None
                        C.grad = None
            return logits.detach(), loss.detach()

        if self.algo == 'fedstructmuon':
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                beta = float(getattr(self.args, 'beta', 0.95))
                score_beta1 = float(
                    getattr(
                        self.args,
                        'struct_score_beta1',
                        getattr(self.args, 'multisub_score_beta1', 0.9),
                    )
                )
                score_beta2 = float(
                    getattr(
                        self.args,
                        'struct_score_beta2',
                        getattr(self.args, 'multisub_score_beta2', 0.999),
                    )
                )
                score_beta1 = float(max(min(score_beta1, 1.0 - 1e-6), 1e-6))
                score_beta2 = float(max(min(score_beta2, 1.0 - 1e-6), 1e-6))
                score_interval = max(
                    int(
                        getattr(
                            self.args,
                            'struct_score_interval',
                            getattr(self.args, 'multisub_score_interval', 1),
                        )
                    ),
                    1,
                )
                with torch.no_grad():
                    for sub_key, X in self.struct_x.items():
                        grad_x = X.grad
                        if grad_x is None:
                            continue
                        if (self._local_step_counter % score_interval) == 0:
                            # AdaMSS-style importance score:
                            # ipt = |X * grad(X)|
                            # exp_avg_ipt <- beta1 * exp_avg_ipt + (1-beta1) * ipt
                            # exp_avg_unc <- beta2 * exp_avg_unc + (1-beta2) * |ipt - exp_avg_ipt|
                            # score = mean(exp_avg_ipt * exp_avg_unc)
                            ipt = (X.data.float() * grad_x.float()).abs()
                            exp_avg_ipt = self.struct_exp_avg_ipt[sub_key]
                            exp_avg_unc = self.struct_exp_avg_unc[sub_key]
                            exp_avg_ipt.mul_(score_beta1).add_(ipt, alpha=(1.0 - score_beta1))
                            exp_avg_unc.mul_(score_beta2).add_((ipt - exp_avg_ipt).abs(), alpha=(1.0 - score_beta2))
                            score_val = float(torch.mean(exp_avg_ipt * exp_avg_unc).item())
                            if np.isfinite(score_val):
                                self.struct_scores[sub_key] = score_val
                        self.struct_m[sub_key].mul_(beta).add_((1.0 - beta) * grad_x)
                        delta_x = zeropower_via_newtonschulz5(self.struct_m[sub_key].float(), self.args.ns_steps)
                        X.data.sub_(self.lr * delta_x.to(dtype=X.data.dtype))
                        X.grad = None
            return logits.detach(), loss.detach()

        if self.algo in ['fedsubadam', 'fedsubsgd']:
            (loss / self.args.n_accum).backward()
            if apply_optim_step:
                if self.optim is not None:
                    self.optim.step()
                    self.optim.zero_grad()
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
