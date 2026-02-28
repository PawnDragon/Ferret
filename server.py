import os
import math
import numpy as np
import torch
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from evaluations import *
from optimizers.ferret_optimizer import FerretFramework
from optimizers.lora_utils import (
    build_lora_model,
    compute_deltaw_from_lora_state,
    extract_lora_state,
    load_lora_state,
    lora_scaling,
    resolve_layer_name_for_model,
)
from optimizers.submuon_utils import init_submuon_state, transport_state
from utils_data.default_tokens import DefaultToken
from utils_data.model_loader import (
    is_qwen3_model,
    maybe_print_qwen3_selfcheck,
    resolve_torch_dtype,
    resolve_model_source,
)


def softmax(vec):
    vec = vec - np.max(vec)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def min_max_norm(vec):
    min_val = np.min(vec)
    return (vec - min_val) / (np.max(vec) + 1e-10 - min_val)


def _as_python_int(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0
        return int(value.item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_finite_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and bool(torch.isfinite(tensor).all().item())


def aggregate_named_adamw_states(named_states_list, weights):
    aggregated_state = {}
    weight_sums = {}
    step_max = {}
    max_exp_avg_sq_weight_sums = {}

    for client_idx, named_state in enumerate(named_states_list):
        if not isinstance(named_state, dict):
            continue
        state_dict = named_state.get('state', {})
        if not isinstance(state_dict, dict):
            continue
        weight = float(weights[client_idx])
        if weight <= 0.0:
            continue

        for param_name, state_entry in state_dict.items():
            if not isinstance(state_entry, dict):
                continue
            exp_avg = state_entry.get('exp_avg', None)
            exp_avg_sq = state_entry.get('exp_avg_sq', None)
            if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                continue
            if (not _is_finite_tensor(exp_avg)) or (not _is_finite_tensor(exp_avg_sq)):
                continue

            if param_name not in aggregated_state:
                aggregated_state[param_name] = {
                    'exp_avg': torch.zeros_like(exp_avg.detach().cpu(), dtype=torch.float32),
                    'exp_avg_sq': torch.zeros_like(exp_avg_sq.detach().cpu(), dtype=torch.float32),
                }
                weight_sums[param_name] = 0.0
                step_max[param_name] = 0

            aggregated_state[param_name]['exp_avg'].add_(exp_avg.detach().cpu().to(dtype=torch.float32), alpha=weight)
            aggregated_state[param_name]['exp_avg_sq'].add_(exp_avg_sq.detach().cpu().to(dtype=torch.float32), alpha=weight)
            weight_sums[param_name] += weight
            step_max[param_name] = max(step_max[param_name], _as_python_int(state_entry.get('step', 0)))

            max_exp_avg_sq = state_entry.get('max_exp_avg_sq', None)
            if isinstance(max_exp_avg_sq, torch.Tensor):
                if 'max_exp_avg_sq' not in aggregated_state[param_name]:
                    aggregated_state[param_name]['max_exp_avg_sq'] = torch.zeros_like(
                        max_exp_avg_sq.detach().cpu(),
                        dtype=torch.float32,
                    )
                    max_exp_avg_sq_weight_sums[param_name] = 0.0
                aggregated_state[param_name]['max_exp_avg_sq'].add_(
                    max_exp_avg_sq.detach().cpu().to(dtype=torch.float32),
                    alpha=weight,
                )
                max_exp_avg_sq_weight_sums[param_name] += weight

    out_state = {'state': {}}
    for param_name, state_entry in aggregated_state.items():
        denom = max(float(weight_sums.get(param_name, 0.0)), 1e-12)
        out_entry = {
            'step': int(step_max.get(param_name, 0)),
            'exp_avg': (state_entry['exp_avg'] / denom).contiguous(),
            'exp_avg_sq': (state_entry['exp_avg_sq'] / denom).contiguous(),
        }
        if 'max_exp_avg_sq' in state_entry:
            max_denom = max(float(max_exp_avg_sq_weight_sums.get(param_name, 0.0)), 1e-12)
            out_entry['max_exp_avg_sq'] = (state_entry['max_exp_avg_sq'] / max_denom).contiguous()
        out_state['state'][param_name] = out_entry
    return out_state


class Server(object):
    def __init__(self, args, eval_loader, candidate_seeds, log_dir):
        self.args = args
        self.eval_loader = eval_loader
        self.candidate_seeds = candidate_seeds
        model_source = resolve_model_source(args.model)
        is_qwen3 = is_qwen3_model(model_source)
        tokenizer_kwargs = {'use_fast': True}
        if is_qwen3:
            tokenizer_kwargs['trust_remote_code'] = True
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, **tokenizer_kwargs)
        self.log_dir = log_dir
        self.algo = getattr(args, 'algo', 'ferret')

        self.tokenizer.model_max_length = self.args.max_length
        special_tokens = dict()
        if self.tokenizer.pad_token is None and (not is_qwen3):
            special_tokens['pad_token'] = DefaultToken.PAD_TOKEN.value
        if self.tokenizer.eos_token is None:
            special_tokens['eos_token'] = DefaultToken.EOS_TOKEN.value
        if self.tokenizer.bos_token is None:
            special_tokens['bos_token'] = DefaultToken.BOS_TOKEN.value
        if self.tokenizer.unk_token is None:
            special_tokens['unk_token'] = DefaultToken.UNK_TOKEN.value
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if is_qwen3:
            maybe_print_qwen3_selfcheck(self.tokenizer, model_source)

        model_dtype = resolve_torch_dtype(getattr(self.args, 'model_dtype', 'bf16'))
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map='cpu',
            torch_dtype=model_dtype,
            trust_remote_code=True,
        )

        self.seed_pool = {seed: 0.0 for seed in self.candidate_seeds}
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.args.device}')
        else:
            self.device = torch.device('cpu')

        # FedSubMuon global states
        self.x_global = {}
        self.m_global = {}
        self.v_global = {}
        self.seeds = {}
        self.submuon_layer_dims = {}
        self.best_metric = math.inf if self.args.eval_metric == 'loss' else -math.inf
        self.seed_rng = np.random.RandomState(self.args.seed + 2026)
        self.global_lora_state = {}
        self.global_deltaW_state = {}
        self.flora_scaling = lora_scaling(self.args)
        self.fedavg_weight_array = None
        self.fedavg_accumulator = {}
        self.fedavg_non_float_state = {}
        self.fedavg_named_optim_accumulator = {}
        self.fedavg_named_optim_weight_sums = {}
        self.fedavg_named_optim_maxsq_weight_sums = {}
        self.fedavg_named_optim_step_max = {}
        self.fedavg_client_count = 0
        self.global_named_optim_state = {'state': {}}
        self._optim_debug_logged = False

        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            self.x_global, self.m_global, self.seeds = init_submuon_state(self.model, self.args.rank_r, self.args.seed)
            if self.algo == 'fedsubsgd':
                self.m_global = {}
            if self.algo == 'fedsubadam':
                self.v_global = {k: torch.zeros_like(v) for k, v in self.x_global.items()}
            name_to_module = dict(self.model.named_modules())
            for layer_name in self.seeds.keys():
                module = name_to_module[layer_name]
                self.submuon_layer_dims[layer_name] = (module.out_features, module.in_features)

        if self.algo == 'fedit':
            init_model = build_lora_model(deepcopy(self.model), self.args)
            self.global_lora_state = extract_lora_state(init_model)
            del init_model

    def _get_ckpt_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)
        return self.log_dir

    def _apply_flora_delta_to_backbone(self):
        if self.algo != 'flora' or len(self.global_deltaW_state) == 0:
            return
        name_to_module = dict(self.model.named_modules())
        with torch.no_grad():
            for layer_name, delta in self.global_deltaW_state.items():
                module = name_to_module.get(layer_name, None)
                if module is None or (not hasattr(module, 'weight')):
                    continue
                delta_tensor = delta.to(device=module.weight.device, dtype=module.weight.dtype)
                if module.weight.shape != delta_tensor.shape:
                    continue
                module.weight.data.add_(delta_tensor * self.flora_scaling)

    def _clone_named_optim_state(self):
        out = {'state': {}}
        if not isinstance(self.global_named_optim_state, dict):
            return out
        state_dict = self.global_named_optim_state.get('state', {})
        if not isinstance(state_dict, dict):
            return out
        for name, state_entry in state_dict.items():
            if not isinstance(state_entry, dict):
                continue
            exp_avg = state_entry.get('exp_avg', None)
            exp_avg_sq = state_entry.get('exp_avg_sq', None)
            if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                continue
            if (not _is_finite_tensor(exp_avg)) or (not _is_finite_tensor(exp_avg_sq)):
                continue
            out_entry = {
                'step': int(_as_python_int(state_entry.get('step', 0))),
                'exp_avg': exp_avg.detach().cpu().clone(),
                'exp_avg_sq': exp_avg_sq.detach().cpu().clone(),
            }
            max_exp_avg_sq = state_entry.get('max_exp_avg_sq', None)
            if isinstance(max_exp_avg_sq, torch.Tensor):
                out_entry['max_exp_avg_sq'] = max_exp_avg_sq.detach().cpu().clone()
            out['state'][name] = out_entry
        return out

    def _maybe_log_global_optim_state(self, tag):
        if self._optim_debug_logged:
            return
        if not isinstance(self.global_named_optim_state, dict):
            return
        state_dict = self.global_named_optim_state.get('state', {})
        if not isinstance(state_dict, dict) or len(state_dict) == 0:
            return
        sampled_name = sorted(state_dict.keys())[0]
        sampled_entry = state_dict[sampled_name]
        exp_avg = sampled_entry.get('exp_avg', None)
        exp_avg_sq = sampled_entry.get('exp_avg_sq', None)
        if isinstance(exp_avg, torch.Tensor) and isinstance(exp_avg_sq, torch.Tensor):
            print(
                f'[debug][server][{tag}] aggregated optim state @ {sampled_name}: '
                f'exp_avg.norm={float(exp_avg.norm().item()):.6e}, '
                f'exp_avg_sq.norm={float(exp_avg_sq.norm().item()):.6e}'
            )
            self._optim_debug_logged = True

    def get_submuon_broadcast_state(self):
        return {
            'x_global': {k: v.clone() for k, v in self.x_global.items()},
            'm_global': {k: v.clone() for k, v in self.m_global.items()} if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
            'v_global': {k: v.clone() for k, v in self.v_global.items()} if self.algo == 'fedsubadam' else None,
            'seeds': dict(self.seeds),
        }

    def get_broadcast_state(self):
        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            return self.get_submuon_broadcast_state()
        if self.algo == 'fedit':
            return {
                'backbone_state_dict': self.model.state_dict(),
                'global_lora_state': self.get_fedit_broadcast_state(),
                'global_named_optim_state': self.get_fedit_broadcast_optim_state(),
            }
        if self.algo == 'fedavg':
            return {
                'backbone_state_dict': self.model.state_dict(),
                'global_named_optim_state': self.get_fedavg_broadcast_optim_state(),
            }
        if self.algo == 'flora':
            return {
                'backbone_state_dict': self.model.state_dict(),
            }
        return {
            'backbone_state_dict': self.model.state_dict(),
        }

    def get_fedit_broadcast_state(self):
        if self.algo != 'fedit':
            return None
        return {k: v.clone() for k, v in self.global_lora_state.items()}

    def get_fedit_broadcast_optim_state(self):
        if self.algo != 'fedit':
            return None
        return self._clone_named_optim_state()

    def get_fedavg_broadcast_state(self):
        if self.algo != 'fedavg':
            return None
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def get_fedavg_broadcast_optim_state(self):
        if self.algo != 'fedavg':
            return None
        return self._clone_named_optim_state()

    def _get_client_weight_array(self, selected_client_list):
        if self.args.equal_weight:
            weight_array = np.array([1.0 for _ in selected_client_list], dtype=np.float64)
            weight_array /= float(len(selected_client_list))
        else:
            weight_array = np.array([len(client.train_loader) for client in selected_client_list], dtype=np.float64)
            weight_array /= float(np.sum(weight_array))
        return weight_array

    def maybe_refresh_submuon_seeds(self, cur_round, force=False):
        if self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            return False
        if (not force) and getattr(self.args, 'stop_F', -1) > 0 and cur_round >= int(self.args.stop_F):
            return False
        if (not force) and self.args.seed_refresh_F <= 0:
            return False
        if (not force) and (cur_round % self.args.seed_refresh_F != 0):
            return False

        old_seeds = dict(self.seeds)
        new_seeds = {}
        for layer_name in self.seeds.keys():
            new_seeds[layer_name] = int(self.seed_rng.randint(1, 2**31 - 1))

        transport_state(
            x_global=self.x_global,
            m_global=self.m_global if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
            old_seeds=old_seeds,
            new_seeds=new_seeds,
            layer_dims=self.submuon_layer_dims,
            rank=self.args.rank_r,
            v_global=self.v_global if self.algo == 'fedsubadam' else None,
        )
        self.seeds = new_seeds
        return True

    def trigger_rebase(self, cur_round):
        return self.maybe_refresh_submuon_seeds(cur_round=cur_round, force=True)

    def aggregate_submuon(self, client_payloads, selected_client_list):
        weight_array = self._get_client_weight_array(selected_client_list)

        new_x = {name: torch.zeros_like(val) for name, val in self.x_global.items()}
        new_m = {name: torch.zeros_like(val) for name, val in self.m_global.items()} if self.algo in ['fedsubmuon', 'fedsubadam'] else None
        new_v = {name: torch.zeros_like(val) for name, val in self.v_global.items()} if self.algo == 'fedsubadam' else None

        for client_idx, payload in enumerate(client_payloads):
            w = float(weight_array[client_idx])
            for name in new_x.keys():
                new_x[name] += payload['x'][name].to(dtype=new_x[name].dtype) * w
                if self.algo in ['fedsubmuon', 'fedsubadam']:
                    new_m[name] += payload['m'][name].to(dtype=new_m[name].dtype) * w
                if self.algo == 'fedsubadam':
                    new_v[name] += payload['v'][name].to(dtype=new_v[name].dtype) * w

        self.x_global = new_x
        if self.algo in ['fedsubmuon', 'fedsubadam']:
            self.m_global = new_m
        if self.algo == 'fedsubadam':
            self.v_global = new_v

    def aggregate_lora(self, client_payloads, selected_client_list):
        if len(client_payloads) == 0:
            return

        weight_array = self._get_client_weight_array(selected_client_list)

        if self.algo == 'fedit':
            state_keys = list(client_payloads[0]['lora_state'].keys())
            new_global_lora = {key: torch.zeros_like(client_payloads[0]['lora_state'][key], dtype=torch.float32) for key in state_keys}

            for client_idx, payload in enumerate(client_payloads):
                w = float(weight_array[client_idx])
                local_state = payload['lora_state']
                for key in state_keys:
                    new_global_lora[key] += local_state[key].to(dtype=torch.float32) * w
            self.global_lora_state = {k: v.cpu() for k, v in new_global_lora.items()}
            client_named_states = [payload.get('named_optim_state', None) for payload in client_payloads]
            self.global_named_optim_state = aggregate_named_adamw_states(client_named_states, weight_array)
            self._maybe_log_global_optim_state(tag='fedit')
            return

        if self.algo == 'flora':
            new_global_delta = {}
            resolved_name_cache = {}
            module_map = dict(self.model.named_modules())
            for client_idx, payload in enumerate(client_payloads):
                w = float(weight_array[client_idx])
                local_delta = compute_deltaw_from_lora_state(payload['lora_state'])
                for layer_name, delta in local_delta.items():
                    if layer_name not in resolved_name_cache:
                        resolved_name_cache[layer_name] = resolve_layer_name_for_model(layer_name, self.model)
                    resolved_layer = resolved_name_cache[layer_name]
                    if resolved_layer not in new_global_delta:
                        new_global_delta[resolved_layer] = torch.zeros_like(delta, dtype=torch.float32)
                    new_global_delta[resolved_layer] += delta.to(dtype=torch.float32) * w

            self.global_deltaW_state = {}
            for layer_name, delta in new_global_delta.items():
                module = module_map.get(layer_name, None)
                target_dtype = module.weight.dtype if (module is not None and hasattr(module, 'weight')) else torch.float32
                self.global_deltaW_state[layer_name] = delta.to(dtype=target_dtype).cpu()
            self._apply_flora_delta_to_backbone()
            return

    def _accumulate_fedavg_named_optim_state(self, named_state, weight):
        if not isinstance(named_state, dict):
            return
        state_dict = named_state.get('state', {})
        if not isinstance(state_dict, dict):
            return

        for param_name, state_entry in state_dict.items():
            if not isinstance(state_entry, dict):
                continue
            exp_avg = state_entry.get('exp_avg', None)
            exp_avg_sq = state_entry.get('exp_avg_sq', None)
            if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                continue

            if param_name not in self.fedavg_named_optim_accumulator:
                self.fedavg_named_optim_accumulator[param_name] = {
                    'exp_avg': torch.zeros_like(exp_avg.detach().cpu(), dtype=torch.float32),
                    'exp_avg_sq': torch.zeros_like(exp_avg_sq.detach().cpu(), dtype=torch.float32),
                }
                self.fedavg_named_optim_weight_sums[param_name] = 0.0
                self.fedavg_named_optim_step_max[param_name] = 0

            self.fedavg_named_optim_accumulator[param_name]['exp_avg'].add_(
                exp_avg.detach().cpu().to(dtype=torch.float32),
                alpha=float(weight),
            )
            self.fedavg_named_optim_accumulator[param_name]['exp_avg_sq'].add_(
                exp_avg_sq.detach().cpu().to(dtype=torch.float32),
                alpha=float(weight),
            )
            self.fedavg_named_optim_weight_sums[param_name] += float(weight)
            self.fedavg_named_optim_step_max[param_name] = max(
                int(self.fedavg_named_optim_step_max[param_name]),
                int(_as_python_int(state_entry.get('step', 0))),
            )

            max_exp_avg_sq = state_entry.get('max_exp_avg_sq', None)
            if isinstance(max_exp_avg_sq, torch.Tensor):
                if 'max_exp_avg_sq' not in self.fedavg_named_optim_accumulator[param_name]:
                    self.fedavg_named_optim_accumulator[param_name]['max_exp_avg_sq'] = torch.zeros_like(
                        max_exp_avg_sq.detach().cpu(),
                        dtype=torch.float32,
                    )
                    self.fedavg_named_optim_maxsq_weight_sums[param_name] = 0.0
                self.fedavg_named_optim_accumulator[param_name]['max_exp_avg_sq'].add_(
                    max_exp_avg_sq.detach().cpu().to(dtype=torch.float32),
                    alpha=float(weight),
                )
                self.fedavg_named_optim_maxsq_weight_sums[param_name] += float(weight)

    def begin_fedavg_aggregation(self, selected_client_list):
        self.fedavg_weight_array = self._get_client_weight_array(selected_client_list)
        self.fedavg_accumulator = {}
        self.fedavg_non_float_state = {}
        self.fedavg_named_optim_accumulator = {}
        self.fedavg_named_optim_weight_sums = {}
        self.fedavg_named_optim_maxsq_weight_sums = {}
        self.fedavg_named_optim_step_max = {}
        self.fedavg_client_count = 0

    def accumulate_fedavg_payload(self, payload, client_idx):
        if self.algo != 'fedavg':
            return
        local_state = payload.get('model_state_dict', None)
        if local_state is None:
            return
        weight = float(self.fedavg_weight_array[client_idx])
        for key, tensor in local_state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            tensor_cpu = tensor.detach().cpu()
            if tensor_cpu.is_floating_point():
                if key not in self.fedavg_accumulator:
                    self.fedavg_accumulator[key] = torch.zeros_like(tensor_cpu, dtype=torch.float32)
                self.fedavg_accumulator[key].add_(tensor_cpu.to(dtype=torch.float32), alpha=weight)
            elif key not in self.fedavg_non_float_state:
                self.fedavg_non_float_state[key] = tensor_cpu.clone()
        self._accumulate_fedavg_named_optim_state(payload.get('named_optim_state', None), weight)
        self.fedavg_client_count += 1

    def finalize_fedavg_aggregation(self):
        if self.algo != 'fedavg' or self.fedavg_client_count == 0:
            return

        current_state = self.model.state_dict()
        averaged_state = {}
        for key, tensor in current_state.items():
            if key in self.fedavg_accumulator:
                averaged_state[key] = self.fedavg_accumulator[key].to(dtype=tensor.dtype)
            elif key in self.fedavg_non_float_state:
                averaged_state[key] = self.fedavg_non_float_state[key]
            else:
                averaged_state[key] = tensor.detach().cpu().clone()

        self.model = self.model.cpu()
        self.model.load_state_dict(averaged_state, strict=True)
        global_named_state = {'state': {}}
        for param_name, state_entry in self.fedavg_named_optim_accumulator.items():
            denom = max(float(self.fedavg_named_optim_weight_sums.get(param_name, 0.0)), 1e-12)
            out_entry = {
                'step': int(self.fedavg_named_optim_step_max.get(param_name, 0)),
                'exp_avg': (state_entry['exp_avg'] / denom).contiguous(),
                'exp_avg_sq': (state_entry['exp_avg_sq'] / denom).contiguous(),
            }
            if 'max_exp_avg_sq' in state_entry:
                max_denom = max(float(self.fedavg_named_optim_maxsq_weight_sums.get(param_name, 0.0)), 1e-12)
                out_entry['max_exp_avg_sq'] = (state_entry['max_exp_avg_sq'] / max_denom).contiguous()
            global_named_state['state'][param_name] = out_entry
        self.global_named_optim_state = global_named_state
        self._maybe_log_global_optim_state(tag='fedavg')
        self.fedavg_weight_array = None
        self.fedavg_accumulator = {}
        self.fedavg_non_float_state = {}
        self.fedavg_named_optim_accumulator = {}
        self.fedavg_named_optim_weight_sums = {}
        self.fedavg_named_optim_maxsq_weight_sums = {}
        self.fedavg_named_optim_step_max = {}
        self.fedavg_client_count = 0

    def aggregate_fedavg(self, client_payloads, selected_client_list):
        if len(client_payloads) == 0:
            return
        self.begin_fedavg_aggregation(selected_client_list)
        for client_idx, payload in enumerate(client_payloads):
            self.accumulate_fedavg_payload(payload, client_idx)
        self.finalize_fedavg_aggregation()

    def aggregate_seed_pool(self, selected_client_list, cur_round=1):
        for seed in self.candidate_seeds:
            self.seed_pool[seed] *= self.args.momentum

        weight_array = self._get_client_weight_array(selected_client_list)
        for client_idx in range(len(selected_client_list)):
            local_seed_pool = selected_client_list[client_idx].local_seed_pool
            for seed, grad in local_seed_pool.items():
                self.seed_pool[seed] += grad * weight_array[client_idx]

        for client in selected_client_list:
            client.clear_model()

    def update_global_model_by_seed_pool(self):
        self.model.to(self.device)

        framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)

        progress_bar = tqdm(range(len(self.seed_pool)))

        # pull the latest model via accumulated {seed, grad} pairs on the server
        for seed, grad in self.seed_pool.items():
            framework.update(seed=seed, grad=grad)
            progress_bar.update(1)
            progress_bar.set_description('server update global model')

        self.model.to('cpu')

    def save_best_submuon_ckpt(self, metric, cur_round):
        if self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd'] or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if not improved:
            return False

        self.best_metric = metric
        ckpt_path = os.path.join(self._get_ckpt_dir(), 'best.pt')
        torch.save(
            {
                'backbone_state_dict': self.model.state_dict(),
                'x_global': {k: v.cpu() for k, v in self.x_global.items()},
                'm_global': {k: v.cpu() for k, v in self.m_global.items()} if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
                'v_global': {k: v.cpu() for k, v in self.v_global.items()} if self.algo == 'fedsubadam' else None,
                'seeds': dict(self.seeds),
                'round': int(cur_round),
                'best_metric': float(metric),
                'hparams': {
                    'algo': self.args.algo,
                    'rank_r': self.args.rank_r,
                    'beta': self.args.beta,
                    'beta1': getattr(self.args, 'beta1', None),
                    'beta2': getattr(self.args, 'beta2', None),
                    'eps': getattr(self.args, 'eps', None),
                    'lr': self.args.lr,
                    'ns_steps': self.args.ns_steps,
                    'seed_refresh_F': self.args.seed_refresh_F,
                },
            },
            ckpt_path,
        )
        print(f'[ckpt] saved to: {ckpt_path}')
        return True

    def save_best_lora_ckpt(self, metric, cur_round):
        if self.algo not in ['fedit', 'flora'] or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if improved:
            self.best_metric = metric

        ckpt_dir = self._get_ckpt_dir()
        ckpt_payload = {
            'algo': self.algo,
            'backbone_state_dict': self.model.state_dict(),
            'round': int(cur_round),
            'best_metric': float(self.best_metric),
            'metric': float(metric),
            'lora_hparams': {
                'lora_r': int(getattr(self.args, 'lora_r', 16)),
                'lora_alpha': float(getattr(self.args, 'lora_alpha', 16.0)),
                'lora_dropout': float(getattr(self.args, 'lora_dropout', 0.0)),
                'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                'lora_bias': getattr(self.args, 'lora_bias', 'none'),
                'scaling': float(self.flora_scaling),
            },
        }

        if self.algo == 'fedit':
            ckpt_payload['global_lora_state'] = {k: v.cpu() for k, v in self.global_lora_state.items()}
        else:
            ckpt_payload['global_deltaW_state'] = {k: v.cpu() for k, v in self.global_deltaW_state.items()}

        final_ckpt_path = os.path.join(ckpt_dir, 'final.pt')
        torch.save(ckpt_payload, final_ckpt_path)
        print(f'[ckpt] saved to: {final_ckpt_path}')
        if improved:
            best_ckpt_path = os.path.join(ckpt_dir, 'best.pt')
            torch.save(ckpt_payload, best_ckpt_path)
            print(f'[ckpt] saved to: {best_ckpt_path}')
        return improved

    def save_best_fedavg_ckpt(self, metric, cur_round):
        if self.algo != 'fedavg' or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if not improved:
            return False

        self.best_metric = metric
        ckpt_path = os.path.join(self._get_ckpt_dir(), 'best.pt')
        torch.save(
            {
                'algo': 'fedavg',
                'backbone_state_dict': self.model.state_dict(),
                'round': int(cur_round),
                'best_metric': float(metric),
                'hparams': {
                    'lr': float(self.args.lr),
                    'adam_beta1': float(getattr(self.args, 'adam_beta1', 0.9)),
                    'adam_beta2': float(getattr(self.args, 'adam_beta2', 0.999)),
                    'adam_eps': float(getattr(self.args, 'adam_eps', 1e-8)),
                    'weight_decay': float(getattr(self.args, 'weight_decay', 0.0)),
                    'batch_or_epoch': self.args.batch_or_epoch,
                    'local_step': int(getattr(self.args, 'local_step', 0)),
                    'n_accum': int(getattr(self.args, 'n_accum', 1)),
                    'max_grad_norm': float(getattr(self.args, 'max_grad_norm', -1.0)),
                },
            },
            ckpt_path,
        )
        print(f'[ckpt] saved to: {ckpt_path}')
        return True

    def eval(self, cur_round, eval_avg_acc):
        if self.args.eval_metric == 'loss':
            eval_metric = self.eval_loss(cur_round)
        else:
            eval_metric = self.eval_generate(cur_round)

        if self.args.save and self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd', 'fedit', 'flora', 'fedavg'] and cur_round > 0:
            save_dir = self.log_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if (self.args.eval_metric == 'loss' and eval_metric < np.min(eval_avg_acc)) or (
                self.args.eval_metric != 'none' and eval_metric > np.max(eval_avg_acc)
            ):
                for file_name in os.listdir(save_dir):
                    if 'best' in file_name:
                        os.remove(os.path.join(save_dir, file_name))
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_best_round{cur_round}.bin'))
            for file_name in os.listdir(save_dir):
                if 'final' in file_name:
                    os.remove(os.path.join(save_dir, file_name))
            torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_final_round{cur_round}.bin'))
        return eval_metric

    def eval_loss(self, cur_round):
        eval_model = None
        framework = None
        temp_eval_model = False
        if self.algo == 'fedit':
            eval_model = build_lora_model(deepcopy(self.model), self.args).to(self.device)
            load_lora_state(eval_model, self.global_lora_state)
            eval_model.eval()
            temp_eval_model = True
        else:
            self.model = self.model.to(self.device)
            self.model.eval()
            eval_model = self.model

        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                self.x_global,
                self.m_global if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
                self.seeds,
                trainable=False,
                v_state=self.v_global if self.algo == 'fedsubadam' else None,
            )

        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        loss_total_eval = 0.0
        num_eval = 0

        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }
                outputs = eval_model(**batch)
                loss = outputs.loss
                progress_bar_eval.update(1)
                if torch.isnan(loss):
                    continue
                loss_total_eval += loss
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, loss: {loss_total_eval / num_eval}')
        print()
        print()

        if framework is not None:
            framework.clear_submuon_state()
        if temp_eval_model:
            eval_model = eval_model.cpu()
            del eval_model
        self.model = self.model.cpu()
        return (loss_total_eval / num_eval).item()

    def eval_generate(self, cur_round):
        eval_model = None
        framework = None
        temp_eval_model = False
        if self.algo == 'fedit':
            eval_model = build_lora_model(deepcopy(self.model), self.args).to(self.device)
            load_lora_state(eval_model, self.global_lora_state)
            eval_model.eval()
            temp_eval_model = True
        else:
            self.model = self.model.to(self.device)
            self.model.eval()
            eval_model = self.model

        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                self.x_global,
                self.m_global if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
                self.seeds,
                trainable=False,
                v_state=self.v_global if self.algo == 'fedsubadam' else None,
            )

        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        acc_total_eval = 0.0
        num_eval = 0

        with torch.inference_mode():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                label_ids = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                output_ids = eval_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_eval += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.tokenizer)
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, metric: {acc_total_eval / num_eval}')
        print()
        print()

        if framework is not None:
            framework.clear_submuon_state()
        if temp_eval_model:
            eval_model = eval_model.cpu()
            del eval_model
        self.model = self.model.cpu()
        return acc_total_eval / num_eval
