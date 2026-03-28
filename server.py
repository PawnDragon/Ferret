import os
import math
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import DataLoader, Subset

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from evaluations import *
from optimizers.ferret_optimizer import FerretFramework
from optimizers.lora_utils import (
    build_lora_model,
    compute_deltaw_from_lora_state,
    extract_classifier_state,
    extract_lora_A_state,
    extract_lora_B_state,
    extract_lora_state,
    get_lora_pair_keys,
    load_classifier_state,
    load_lora_A_state,
    load_lora_B_state,
    load_lora_state,
    initialize_federa_lora_state,
    lora_scaling,
    resolve_layer_name_for_model,
)
from optimizers.florg_utils import (
    build_florg_model,
    extract_florg_A_state,
    extract_florg_basis_state,
    extract_florg_seed_state,
    load_florg_A_state,
    sample_florg_delta_norm,
)
from optimizers.fedmultisub_utils import initialize_subspaces, select_topk_subspaces
from optimizers.fedstruct_utils import initialize_struct_subspaces
from optimizers.submuon_utils import fold_submuon_core_into_backbone, init_submuon_state, make_uv, transport_state
from utils_data.default_tokens import DefaultToken
from utils_data.gsm8k_metrics import compute_gsm8k_metrics
from utils_data.math_metrics import (
    compute_math_metrics,
    extract_math_gold_final_answer,
    extract_math_pred_final_answer,
)
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


def sanitize_greedy_generation_config(model):
    """
    Normalize generation_config to greedy defaults to avoid repeated warnings
    about sampling-only flags when do_sample=False.
    """
    gen_cfg = getattr(model, 'generation_config', None)
    if gen_cfg is None:
        return
    try:
        gen_cfg.do_sample = False
        gen_cfg.num_beams = 1
        if hasattr(gen_cfg, 'temperature'):
            gen_cfg.temperature = 1.0
        if hasattr(gen_cfg, 'top_p'):
            gen_cfg.top_p = 1.0
        if hasattr(gen_cfg, 'top_k'):
            gen_cfg.top_k = 50
    except Exception:
        return


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
        self.round_eval_sample = float(getattr(args, 'round_eval_sample', 1.0))
        self.round_eval_rng = np.random.RandomState(int(getattr(args, 'seed', 42)) + 2027)
        self._round_eval_sample_logged = False
        self._fixed_round_eval_loader = None
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
        self.model_dtype = model_dtype
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
        self.u_global = {}
        self.v_basis_global = {}
        self.submuon_layer_dims = {}
        self._submuonv2_fold_logged = False
        self.best_metric = math.inf if self.args.eval_metric == 'loss' else -math.inf
        self.seed_rng = np.random.RandomState(self.args.seed + 2026)
        self.global_lora_state = {}
        self.global_lora_A_state = {}
        self.global_lora_B_state = {}
        self.global_classifier_state = {}
        self.global_deltaW_state = {}
        self.global_florg_A_state = {}
        self.global_florg_seed_state = {}
        self.global_florg_basis_state = {}
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
        self._fedex_debug_logged = False
        self._florg_debug_logged = False
        self._florg_eval_debug_logged = False
        self._multisub_debug_logged = False
        self.global_multisub_metadata = {}
        self.global_multisub_b_state = {}
        self.global_multisub_c_state = {}
        self.global_multisub_scores = {}
        self.global_multisub_selected_keys = []
        self.global_struct_metadata = {}
        self.global_struct_x_state = {}
        self.global_struct_scores = {}
        self.global_struct_selected_keys = []
        self._struct_debug_logged = False
        self._struct_topk_schedule_logged = False

        if self.algo in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd']:
            self.x_global, self.m_global, self.seeds = init_submuon_state(
                self.model,
                self.args.rank_r,
                self.args.seed,
                raw_target_modules=getattr(self.args, 'lora_target_modules', None),
            )
            if self.algo != 'fedsubmuon':
                self.m_global = {}
                self.v_global = {}
            name_to_module = dict(self.model.named_modules())
            for layer_name in self.seeds.keys():
                module = name_to_module[layer_name]
                self.submuon_layer_dims[layer_name] = (module.out_features, module.in_features)
            if self.algo == 'fedsubmuon_gt':
                self.u_global = {}
                self.v_basis_global = {}
                basis_init_mode = self._resolve_gt_basis_init_mode()
                svd_init_logs = []
                for layer_name, (out_dim, in_dim) in self.submuon_layer_dims.items():
                    module = name_to_module[layer_name]
                    U, V, svd_meta = self._initialize_submuon_gt_basis(
                        layer_name=layer_name,
                        module=module,
                        seed=int(self.seeds[layer_name]),
                        out_dim=int(out_dim),
                        in_dim=int(in_dim),
                        basis_init_mode=basis_init_mode,
                    )
                    self.u_global[layer_name] = U.cpu().contiguous()
                    self.v_basis_global[layer_name] = V.cpu().contiguous()
                    if isinstance(svd_meta, dict):
                        svd_init_logs.append(svd_meta)
                if basis_init_mode != 'random' and bool(getattr(self.args, 'log', False)):
                    max_u_orth = float(np.max([item['u_orth_err'] for item in svd_init_logs])) if len(svd_init_logs) > 0 else 0.0
                    max_v_orth = float(np.max([item['v_orth_err'] for item in svd_init_logs])) if len(svd_init_logs) > 0 else 0.0
                    print(
                        f'[fedsubmuon_gt][init] basis_init_mode={basis_init_mode} layers={len(svd_init_logs)} '
                        f'max_u_orth_err={max_u_orth:.6e} max_v_orth_err={max_v_orth:.6e}'
                    )
                    for item in svd_init_logs[:3]:
                        print(
                            f'[fedsubmuon_gt][init] layer={item["layer"]} '
                            f'mode={item.get("basis_init_mode", basis_init_mode)} '
                            f'rank={item["rank"]} sigma_max={item["sigma_max"]:.6e} sigma_min={item["sigma_min"]:.6e}'
                        )

        if self.algo == 'fedit':
            init_model = build_lora_model(deepcopy(self.model), self.args)
            self.global_lora_state = extract_lora_state(init_model)
            del init_model
        if self.algo == 'federa':
            self.global_lora_state, federa_logs = initialize_federa_lora_state(self.model, self.args)
            if bool(getattr(self.args, 'log', False)):
                total_layers = len(federa_logs)
                total_delta_norm = float(np.sum([item['delta_norm'] for item in federa_logs])) if total_layers > 0 else 0.0
                max_recon_err = float(np.max([item['recon_err'] for item in federa_logs])) if total_layers > 0 else 0.0
                print(
                    f'[federa][init] layers={total_layers}, '
                    f'delta_norm_sum={total_delta_norm:.6e}, max_recon_err={max_recon_err:.6e}'
                )
                for item in federa_logs[:3]:
                    print(
                        f'[federa][init] layer={item["layer"]} rank={item["rank"]} '
                        f'rank_eff={item["rank_eff"]} delta_norm={item["delta_norm"]:.6e} '
                        f'recon_err={item["recon_err"]:.6e}'
                    )
        if self.algo == 'fedsalora':
            init_model = build_lora_model(deepcopy(self.model), self.args)
            self.global_lora_A_state = extract_lora_A_state(init_model)
            del init_model
        if self.algo == 'fedexlora':
            init_model = build_lora_model(deepcopy(self.model), self.args)
            self.global_lora_A_state = extract_lora_A_state(init_model)
            self.global_lora_B_state = extract_lora_B_state(init_model)
            self.global_classifier_state = extract_classifier_state(init_model)
            load_classifier_state(self.model, self.global_classifier_state)
            del init_model
        if self.algo == 'florg':
            init_model = build_florg_model(deepcopy(self.model), self.args)
            self.global_florg_A_state = extract_florg_A_state(init_model)
            self.global_florg_seed_state = extract_florg_seed_state(init_model)
            self.global_florg_basis_state = extract_florg_basis_state(init_model)
            del init_model
        if self.algo == 'fedmultisubmuon':
            base_seed = int(getattr(self.args, 'multisub_seed_base', int(self.args.seed) + 13579))
            (
                self.global_multisub_metadata,
                self.global_multisub_b_state,
                self.global_multisub_c_state,
                self.global_multisub_scores,
            ) = initialize_subspaces(
                self.model,
                rank_r=int(self.args.rank_r),
                svd_rank=int(getattr(self.args, 'svd_rank', 500)),
                num_subspaces=int(getattr(self.args, 'multisub_num_subspaces', 4)),
                base_seed=base_seed,
                target_modules=getattr(self.args, 'lora_target_modules', None),
            )
            # Warmup policy:
            # round-1 uses all subspaces so every subspace can collect score at least once.
            # from round-2 onward, top-k is selected in aggregate_fedmultisubmuon() using round-(t-1) scores.
            self.global_multisub_selected_keys = list(self.global_multisub_b_state.keys())
            print(
                f'[fedmultisubmuon] initialized {len(self.global_multisub_b_state)} subspaces; '
                f'round-1 warmup uses all={len(self.global_multisub_selected_keys)}'
            )
        if self.algo == 'fedstructmuon':
            base_seed = int(getattr(self.args, 'struct_seed_base', int(self.args.seed) + 24680))
            (
                self.global_struct_metadata,
                self.global_struct_x_state,
                self.global_struct_scores,
            ) = initialize_struct_subspaces(
                self.model,
                rank_r=int(self.args.rank_r),
                rank_left=int(getattr(self.args, 'rank_left', int(self.args.rank_r))),
                rank_right=int(getattr(self.args, 'rank_right', int(self.args.rank_r))),
                svd_rank=int(getattr(self.args, 'svd_rank', 500)),
                num_subspaces=int(getattr(self.args, 'struct_num_subspaces', 4)),
                base_seed=base_seed,
                target_modules=getattr(self.args, 'lora_target_modules', None),
            )
            # Warmup policy:
            # round-1 uses all subspaces so every subspace can collect score at least once.
            self.global_struct_selected_keys = list(self.global_struct_x_state.keys())
            print(
                f'[fedstructmuon] initialized {len(self.global_struct_x_state)} subspaces; '
                f'round-1 warmup uses all={len(self.global_struct_selected_keys)}'
            )

    def _build_round_eval_loader(self, cur_round):
        ratio = float(getattr(self.args, 'round_eval_sample', 1.0))
        if ratio >= 1.0:
            return self.eval_loader
        if self._fixed_round_eval_loader is not None:
            return self._fixed_round_eval_loader
        if self.eval_loader is None:
            return self.eval_loader
        dataset = getattr(self.eval_loader, 'dataset', None)
        if dataset is None:
            return self.eval_loader
        total = int(len(dataset))
        if total <= 0:
            return self.eval_loader

        sample_n = int(total * ratio)
        if ratio <= 0.0:
            sample_n = 1
        else:
            sample_n = max(sample_n, 1)
        sample_n = min(sample_n, total)
        if sample_n >= total:
            return self.eval_loader

        indices = self.round_eval_rng.choice(total, size=sample_n, replace=False).tolist()
        if bool(getattr(self.args, 'log', False)) and (not self._round_eval_sample_logged):
            print(
                f'[info] round eval sampling enabled: ratio={ratio:.4f}, '
                f'sampled={sample_n}/{total}'
            )
            self._round_eval_sample_logged = True
        subset = Subset(dataset, indices=indices)
        self._fixed_round_eval_loader = DataLoader(
            subset,
            shuffle=False,
            batch_size=getattr(self.eval_loader, 'batch_size', 1),
            collate_fn=getattr(self.eval_loader, 'collate_fn', None),
        )
        return self._fixed_round_eval_loader

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

    def _get_comm_optim_state_dtype(self):
        if self.model_dtype in [torch.float16, torch.bfloat16]:
            return self.model_dtype
        return None

    def _clone_named_optim_state(self, comm_compress=False):
        out = {'state': {}}
        if not isinstance(self.global_named_optim_state, dict):
            return out
        state_dict = self.global_named_optim_state.get('state', {})
        if not isinstance(state_dict, dict):
            return out
        target_dtype = self._get_comm_optim_state_dtype() if comm_compress else None
        for name, state_entry in state_dict.items():
            if not isinstance(state_entry, dict):
                continue
            exp_avg = state_entry.get('exp_avg', None)
            exp_avg_sq = state_entry.get('exp_avg_sq', None)
            if not isinstance(exp_avg, torch.Tensor) or not isinstance(exp_avg_sq, torch.Tensor):
                continue
            if (not _is_finite_tensor(exp_avg)) or (not _is_finite_tensor(exp_avg_sq)):
                continue
            exp_avg_out = exp_avg.detach().cpu()
            exp_avg_sq_out = exp_avg_sq.detach().cpu()
            if target_dtype is not None:
                exp_avg_out = exp_avg_out.to(dtype=target_dtype)
                exp_avg_sq_out = exp_avg_sq_out.to(dtype=target_dtype)
            out_entry = {
                'step': int(_as_python_int(state_entry.get('step', 0))),
                'exp_avg': exp_avg_out.clone(),
                'exp_avg_sq': exp_avg_sq_out.clone(),
            }
            max_exp_avg_sq = state_entry.get('max_exp_avg_sq', None)
            if isinstance(max_exp_avg_sq, torch.Tensor):
                max_exp_avg_sq_out = max_exp_avg_sq.detach().cpu()
                if target_dtype is not None:
                    max_exp_avg_sq_out = max_exp_avg_sq_out.to(dtype=target_dtype)
                out_entry['max_exp_avg_sq'] = max_exp_avg_sq_out.clone()
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
        aggregate_muon_state = bool(self.algo == 'fedsubmuon' and getattr(self.args, 'aggregate_muon_state', False))
        return {
            'x_global': {k: v.clone() for k, v in self.x_global.items()},
            'm_global': {k: v.clone() for k, v in self.m_global.items()} if aggregate_muon_state else None,
            'v_global': None,
            'seeds': dict(self.seeds),
        }

    def get_submuonv2_broadcast_state(self):
        return {
            'x_global': {k: v.clone() for k, v in self.x_global.items()},
            'm_global': None,
            'v_global': None,
            'seeds': dict(self.seeds),
        }

    def is_submuon_gt_refresh_round(self, cur_round, force=False):
        if self.algo != 'fedsubmuon_gt':
            return False
        if (not force) and getattr(self.args, 'stop_F', -1) > 0 and cur_round >= int(self.args.stop_F):
            return False
        if (not force) and int(getattr(self.args, 'seed_refresh_F', 0)) <= 0:
            return False
        if (not force) and (int(cur_round) % int(self.args.seed_refresh_F) != 0):
            return False
        return True

    def get_submuon_gt_broadcast_state(self, is_refresh_round=False):
        return {
            'x_global': {k: v.clone() for k, v in self.x_global.items()},
            'm_global': None,
            'v_global': None,
            'seeds': dict(self.seeds),
            'u_global': {k: v.clone() for k, v in self.u_global.items()},
            'v_basis_global': {k: v.clone() for k, v in self.v_basis_global.items()},
            'is_refresh_round': bool(is_refresh_round),
        }

    def get_broadcast_state(self):
        if self.algo == 'fedsubmuonv2':
            return self.get_submuonv2_broadcast_state()
        if self.algo == 'fedsubmuon_gt':
            return self.get_submuon_gt_broadcast_state(is_refresh_round=False)
        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            return self.get_submuon_broadcast_state()
        if self.algo == 'fedmultisubmuon':
            return self.get_multisub_broadcast_state()
        if self.algo == 'fedstructmuon':
            return self.get_struct_broadcast_state()
        if self.algo in ['fedit', 'federa']:
            return {
                'backbone_state_dict': self.model.state_dict(),
                'global_lora_state': self.get_fedit_broadcast_state(),
            }
        if self.algo == 'fedavg':
            return {
                'backbone_state_dict': self.model.state_dict(),
            }
        if self.algo == 'flora':
            return {
                'backbone_state_dict': self.model.state_dict(),
            }
        if self.algo == 'fedsalora':
            return {
                'backbone_state_dict': self.model.state_dict(),
                'global_lora_A_state': self.get_fedsalora_broadcast_state(),
            }
        if self.algo == 'fedexlora':
            return self.get_broadcast_state_fedexlora()
        if self.algo == 'florg':
            return {
                'global_florg_A_state': self.get_florg_broadcast_state(),
            }
        return {
            'backbone_state_dict': self.model.state_dict(),
        }

    def get_multisub_broadcast_state(self):
        if self.algo != 'fedmultisubmuon':
            return None
        selected_keys = list(self.global_multisub_selected_keys)
        metadata = {}
        b_global = {}
        c_global = {}
        score_state = {}
        for key in selected_keys:
            if (
                key not in self.global_multisub_metadata
                or key not in self.global_multisub_b_state
                or key not in self.global_multisub_c_state
            ):
                continue
            meta = self.global_multisub_metadata[key]
            metadata[key] = {
                'layer_name': str(meta['layer_name']),
                'indices': meta['indices'].clone(),
                'A': meta['A'].clone(),
                'rank_big': int(meta.get('rank_big', int(self.global_multisub_b_state[key].shape[0]))),
                'rank_small': int(meta.get('rank_small', int(self.global_multisub_c_state[key].shape[0]))),
                'flat_id': int(meta.get('flat_id', -1)),
            }
            b_global[key] = self.global_multisub_b_state[key].clone()
            c_global[key] = self.global_multisub_c_state[key].clone()
            score_state[key] = float(self.global_multisub_scores.get(key, 0.0))
        return {
            'b_global': b_global,
            'c_global': c_global,
            'metadata': metadata,
            'selected_keys': selected_keys,
            'score_state': score_state,
        }

    def get_struct_broadcast_state(self):
        if self.algo != 'fedstructmuon':
            return None
        selected_keys = list(self.global_struct_selected_keys)
        metadata = {}
        x_global = {}
        score_state = {}
        for key in selected_keys:
            if key not in self.global_struct_metadata or key not in self.global_struct_x_state:
                continue
            meta = self.global_struct_metadata[key]
            metadata[key] = {
                'layer_name': str(meta['layer_name']),
                'indices': meta['indices'].clone(),
                'A': meta['A'].clone(),
                'V': meta['V'].clone(),
                'rank': int(meta.get('rank', int(self.global_struct_x_state[key].shape[0]))),
                'rank_left': int(meta.get('rank_left', int(self.global_struct_x_state[key].shape[0]))),
                'rank_right': int(meta.get('rank_right', int(self.global_struct_x_state[key].shape[1]))),
                'flat_id': int(meta.get('flat_id', -1)),
            }
            x_global[key] = self.global_struct_x_state[key].clone()
            score_state[key] = float(self.global_struct_scores.get(key, 0.0))
        return {
            'x_global': x_global,
            'metadata': metadata,
            'selected_keys': selected_keys,
            'score_state': score_state,
        }

    def log_multisub_selection(self, cur_round, broadcast_state=None):
        if self.algo != 'fedmultisubmuon':
            return
        state = broadcast_state if isinstance(broadcast_state, dict) else self.get_multisub_broadcast_state()
        if not isinstance(state, dict):
            print(f'[fedmultisubmuon][round {cur_round}] selected subspaces: 0')
            return

        selected_keys = state.get('selected_keys', [])
        metadata = state.get('metadata', {})
        score_state = state.get('score_state', {})
        if not isinstance(selected_keys, (list, tuple)):
            selected_keys = []
        print(f'[fedmultisubmuon][round {cur_round}] selected subspaces: {len(selected_keys)}')

        round_scores = []
        for idx, sub_key in enumerate(selected_keys):
            meta = metadata.get(sub_key, {}) if isinstance(metadata, dict) else {}
            layer_name = str(meta.get('layer_name', 'unknown'))
            rank_big = int(meta.get('rank_big', -1))
            rank_small = int(meta.get('rank_small', -1))
            idx_tensor = meta.get('indices', None)
            n_cols = int(idx_tensor.numel()) if isinstance(idx_tensor, torch.Tensor) else -1
            if isinstance(score_state, dict) and sub_key in score_state:
                score_val = float(score_state[sub_key])
            else:
                score_val = float(self.global_multisub_scores.get(sub_key, float('nan')))
            round_scores.append(score_val)
            print(
                f'  [{idx}] key={sub_key}, layer={layer_name}, '
                f'rank_big={rank_big}, rank_small={rank_small}, cols={n_cols}, score={score_val:.6e}'
            )
        finite_scores = [s for s in round_scores if np.isfinite(s)]
        if len(finite_scores) > 0:
            print(
                f'[fedmultisubmuon][round {cur_round}] score summary: '
                f'min={min(finite_scores):.6e}, max={max(finite_scores):.6e}, '
                f'mean={float(np.mean(finite_scores)):.6e}'
            )

    def log_struct_selection(self, cur_round, broadcast_state=None):
        if self.algo != 'fedstructmuon':
            return
        state = broadcast_state if isinstance(broadcast_state, dict) else self.get_struct_broadcast_state()
        if not isinstance(state, dict):
            print(f'[fedstructmuon][round {cur_round}] selected subspaces: 0')
            return

        selected_keys = state.get('selected_keys', [])
        if not isinstance(selected_keys, (list, tuple)):
            selected_keys = []
        print(f'[fedstructmuon][round {cur_round}] selected subspaces: {len(selected_keys)}')

    def get_fedit_broadcast_state(self):
        if self.algo not in ['fedit', 'federa']:
            return None
        return {k: v.clone() for k, v in self.global_lora_state.items()}

    def get_fedit_broadcast_optim_state(self):
        if self.algo != 'fedit':
            return None
        return self._clone_named_optim_state(comm_compress=True)

    def get_fedsalora_broadcast_state(self):
        if self.algo != 'fedsalora':
            return None
        return {k: v.clone() for k, v in self.global_lora_A_state.items()}

    def get_broadcast_state_fedexlora(self):
        if self.algo != 'fedexlora':
            return None
        return {
            'backbone_state_dict': self.model.state_dict(),
            'global_lora_A_state': {k: v.clone() for k, v in self.global_lora_A_state.items()},
            'global_lora_B_state': {k: v.clone() for k, v in self.global_lora_B_state.items()},
            'global_classifier_state': {k: v.clone() for k, v in self.global_classifier_state.items()},
        }

    def get_florg_broadcast_state(self):
        if self.algo != 'florg':
            return None
        return {k: v.clone() for k, v in self.global_florg_A_state.items()}

    def get_florg_seed_state(self):
        if self.algo != 'florg':
            return None
        seed_state = self.global_florg_seed_state
        if not isinstance(seed_state, dict):
            return {'base_seed': None, 'layer_seeds': {}}
        layer_seeds = seed_state.get('layer_seeds', {})
        if not isinstance(layer_seeds, dict):
            layer_seeds = {}
        return {
            'base_seed': seed_state.get('base_seed', None),
            'layer_seeds': {str(k): int(v) for k, v in layer_seeds.items()},
        }

    def get_florg_basis_state(self):
        if self.algo != 'florg':
            return None
        out = {}
        for layer_name, layer_state in self.global_florg_basis_state.items():
            if not isinstance(layer_state, dict):
                continue
            l_tensor = layer_state.get('L', None)
            r_tensor = layer_state.get('R', None)
            if not isinstance(l_tensor, torch.Tensor) or not isinstance(r_tensor, torch.Tensor):
                continue
            out[layer_name] = {
                'L': l_tensor.clone(),
                'R': r_tensor.clone(),
            }
        return out

    def get_florg_basis_state_ref(self):
        if self.algo != 'florg':
            return None
        return self.global_florg_basis_state

    def get_fedavg_broadcast_state(self):
        if self.algo != 'fedavg':
            return None
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    def get_fedavg_broadcast_optim_state(self):
        if self.algo != 'fedavg':
            return None
        return self._clone_named_optim_state(comm_compress=True)

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

        aggregate_muon_state = bool(self.algo == 'fedsubmuon' and getattr(self.args, 'aggregate_muon_state', False))
        transport_state(
            x_global=self.x_global,
            m_global=self.m_global if aggregate_muon_state else None,
            old_seeds=old_seeds,
            new_seeds=new_seeds,
            layer_dims=self.submuon_layer_dims,
            rank=self.args.rank_r,
            v_global=None,
        )
        self.seeds = new_seeds
        return True

    def maybe_refresh_submuonv2_seeds(self, cur_round, force=False):
        if self.algo != 'fedsubmuonv2':
            return False
        if (not force) and getattr(self.args, 'stop_F', -1) > 0 and cur_round >= int(self.args.stop_F):
            return False
        if (not force) and self.args.seed_refresh_F <= 0:
            return False
        if (not force) and (cur_round % self.args.seed_refresh_F != 0):
            return False

        old_seeds = dict(self.seeds)
        old_x_snapshot = {name: tensor.detach().cpu().clone() for name, tensor in self.x_global.items()}
        fold_stats = fold_submuon_core_into_backbone(
            model=self.model,
            x_state=old_x_snapshot,
            seeds=old_seeds,
            rank=self.args.rank_r,
            layer_dims=self.submuon_layer_dims,
        )
        self.x_global = {name: torch.zeros_like(tensor) for name, tensor in self.x_global.items()}
        max_abs = 0.0
        for tensor in self.x_global.values():
            if tensor.numel() > 0:
                max_abs = max(max_abs, float(torch.max(torch.abs(tensor)).item()))
        if max_abs != 0.0:
            raise RuntimeError(f'[fedsubmuonv2] server-side X reset failed, max_abs={max_abs}')

        new_seeds = {}
        for layer_name in self.seeds.keys():
            new_seeds[layer_name] = int(self.seed_rng.randint(1, 2**31 - 1))

        self.seeds = new_seeds
        if (not self._submuonv2_fold_logged) and fold_stats.get('num_layers', 0) > 0:
            refresh_round = int(cur_round)
            sample_layer = next(iter(old_seeds.keys())) if len(old_seeds) > 0 else 'n/a'
            old_seed = old_seeds.get(sample_layer, None) if sample_layer != 'n/a' else None
            new_seed = self.seeds.get(sample_layer, None) if sample_layer != 'n/a' else None
            print(
                f'[fedsubmuonv2][server] refresh_round={refresh_round} '
                f'old_seed={old_seed} new_seed={new_seed} '
                f'folded_layers={int(fold_stats["num_layers"])} '
                f'delta_norm={float(fold_stats["delta_norm"]):.6e}'
            )
            self._submuonv2_fold_logged = True
        return True

    def _orthonormalize_columns(self, mat):
        if not isinstance(mat, torch.Tensor) or mat.ndim != 2:
            raise RuntimeError('[fedsubmuon_gt] orthonormalize expects a 2D tensor')
        q_mat, _ = torch.linalg.qr(mat.float(), mode='reduced')
        return q_mat.contiguous()

    def _rank1_approx_tangent(self, tangent):
        if (not isinstance(tangent, torch.Tensor)) or tangent.ndim != 2:
            raise RuntimeError('[fedsubmuon_gt] rank-1 approximation expects a 2D tensor')
        tangent_f = tangent.float()
        if tangent_f.numel() == 0:
            return torch.zeros_like(tangent_f), 0.0
        tangent_norm = float(torch.linalg.norm(tangent_f).item())
        if tangent_norm <= 0.0 or (not np.isfinite(tangent_norm)):
            return torch.zeros_like(tangent_f), 0.0
        U_svd, S_svd, Vh_svd = torch.linalg.svd(tangent_f, full_matrices=False)
        if int(S_svd.numel()) == 0:
            return torch.zeros_like(tangent_f), 0.0
        sigma_top = float(S_svd[0].item())
        if (not np.isfinite(sigma_top)) or sigma_top <= 0.0:
            return torch.zeros_like(tangent_f), 0.0
        rank1 = (S_svd[0] * (U_svd[:, :1] @ Vh_svd[:1, :])).contiguous()
        return rank1, sigma_top

    def _resolve_gt_effective_step(self, basis, tangent_used, sigma_top, rank1_enabled, gt_sub_lr, gt_target_rel_step, eps=1e-8):
        tau_rel = float(gt_target_rel_step)
        if tau_rel <= 0.0:
            return float(gt_sub_lr)
        basis_norm = float(torch.linalg.norm(basis.float()).item())
        if rank1_enabled:
            denom = float(sigma_top)
        else:
            denom = float(torch.linalg.norm(tangent_used.float()).item())
        if (not np.isfinite(denom)) or denom <= float(eps):
            return 0.0
        return float(tau_rel * basis_norm / (denom + float(eps)))

    def _resolve_gt_basis_init_mode(self):
        mode = str(getattr(self.args, 'basis_init_mode', 'random')).lower()
        valid_modes = {'random', 'svd_left', 'svd_right', 'svd_both'}
        if mode not in valid_modes:
            raise RuntimeError(f'[fedsubmuon_gt] invalid basis_init_mode={mode}')
        return mode

    def _resolve_gt_update_mode(self):
        mode = str(getattr(self.args, 'gt_update_mode', 'both')).lower()
        valid_modes = {'both', 'left', 'right', 'alternate_lr', 'alternate_rl'}
        if mode not in valid_modes:
            raise RuntimeError(f'[fedsubmuon_gt] invalid gt_update_mode={mode}')
        return mode

    def _compute_gt_refresh_index(self, cur_round):
        refresh_f = int(getattr(self.args, 'seed_refresh_F', 0))
        if refresh_f <= 0:
            return 0
        effective_round = int(max(cur_round, 0))
        stop_f = int(getattr(self.args, 'stop_F', -1))
        if stop_f > 0:
            effective_round = min(effective_round, stop_f - 1)
        if effective_round <= 0:
            return 0
        return int(effective_round // refresh_f)

    def _resolve_gt_refresh_side(self, cur_round):
        mode = self._resolve_gt_update_mode()
        refresh_idx = int(self._compute_gt_refresh_index(cur_round))
        if mode == 'both':
            return True, True, 'both', refresh_idx, mode
        if mode == 'left':
            return True, False, 'left', refresh_idx, mode
        if mode == 'right':
            return False, True, 'right', refresh_idx, mode
        if refresh_idx <= 0:
            refresh_idx = 1
        if mode == 'alternate_lr':
            if (refresh_idx % 2) == 1:
                return True, False, 'left', refresh_idx, mode
            return False, True, 'right', refresh_idx, mode
        if mode == 'alternate_rl':
            if (refresh_idx % 2) == 1:
                return False, True, 'right', refresh_idx, mode
            return True, False, 'left', refresh_idx, mode
        raise RuntimeError(f'[fedsubmuon_gt] unsupported gt_update_mode={mode}')

    def _compute_submuon_gt_svd_basis(self, layer_name, module, rank):
        if module is None or (not hasattr(module, 'weight')):
            raise RuntimeError(f'[fedsubmuon_gt] SVD init cannot find target weight for layer={layer_name}')
        weight = module.weight.detach().to(device='cpu', dtype=torch.float32)
        if weight.ndim != 2:
            raise RuntimeError(
                f'[fedsubmuon_gt] SVD init expects a 2D weight at {layer_name}, got shape={tuple(weight.shape)}'
            )
        rank_eff = int(min(rank, int(weight.shape[0]), int(weight.shape[1])))
        if rank_eff <= 0:
            raise RuntimeError(
                f'[fedsubmuon_gt] invalid SVD init rank for {layer_name}: '
                f'rank_r={rank}, weight_shape={tuple(weight.shape)}'
            )
        U_full, S_full, Vh_full = torch.linalg.svd(weight, full_matrices=False)
        U = U_full[:, :rank_eff].contiguous()
        V = Vh_full[:rank_eff, :].t().contiguous()
        if rank_eff != rank:
            raise RuntimeError(
                f'[fedsubmuon_gt] SVD init rank mismatch at {layer_name}: '
                f'rank_r={rank}, rank_eff={rank_eff}. Please reduce --rank_r for this model.'
            )
        eye_u = torch.eye(rank_eff, dtype=torch.float32)
        eye_v = torch.eye(rank_eff, dtype=torch.float32)
        u_orth_err = torch.linalg.norm(U.t() @ U - eye_u)
        v_orth_err = torch.linalg.norm(V.t() @ V - eye_v)
        meta = {
            'layer': layer_name,
            'rank': int(rank_eff),
            'sigma_max': float(S_full[0].item()) if int(S_full.numel()) > 0 else 0.0,
            'sigma_min': float(S_full[rank_eff - 1].item()) if int(S_full.numel()) >= rank_eff else 0.0,
            'u_orth_err': float(u_orth_err.item()),
            'v_orth_err': float(v_orth_err.item()),
        }
        return U.cpu().contiguous(), V.cpu().contiguous(), meta

    def _initialize_submuon_gt_basis(self, layer_name, module, seed, out_dim, in_dim, basis_init_mode=None):
        rank = int(self.args.rank_r)
        mode = str(basis_init_mode or self._resolve_gt_basis_init_mode()).lower()
        U_rand, V_rand = make_uv(
            seed=int(seed),
            out_dim=int(out_dim),
            in_dim=int(in_dim),
            r=rank,
            device='cpu',
            dtype=torch.float32,
        )
        if mode == 'random':
            return U_rand.cpu().contiguous(), V_rand.cpu().contiguous(), None

        U_svd, V_svd, svd_meta = self._compute_submuon_gt_svd_basis(
            layer_name=layer_name,
            module=module,
            rank=rank,
        )
        if mode == 'svd_both':
            U_out, V_out = U_svd, V_svd
        elif mode == 'svd_left':
            U_out, V_out = U_svd, V_rand
        elif mode == 'svd_right':
            U_out, V_out = U_rand, V_svd
        else:
            raise RuntimeError(f'[fedsubmuon_gt] invalid basis_init_mode={mode}')

        if isinstance(svd_meta, dict):
            svd_meta['basis_init_mode'] = str(mode)
        return U_out.cpu().contiguous(), V_out.cpu().contiguous(), svd_meta

    def _merge_layer_residual_to_backbone(self, layer_name, residual):
        module = dict(self.model.named_modules()).get(layer_name, None)
        if module is None or (not hasattr(module, 'weight')):
            raise RuntimeError(f'[fedsubmuon_gt] cannot find target layer for residual merge: {layer_name}')
        if tuple(module.weight.shape) != tuple(residual.shape):
            raise RuntimeError(
                f'[fedsubmuon_gt] residual shape mismatch at {layer_name}: '
                f'residual={tuple(residual.shape)}, weight={tuple(module.weight.shape)}'
            )
        module.weight.data.add_(residual.to(device=module.weight.device, dtype=module.weight.dtype))

    def aggregate_submuon_gt(self, client_payloads, selected_client_list, cur_round, is_refresh_round):
        gt_topk = int(getattr(self.args, 'gt_topk', 0))
        gt_rank1_approx = bool(getattr(self.args, 'gt_rank1_approx', False))
        gt_target_rel_step = float(getattr(self.args, 'gt_target_rel_step', 0.0))
        if self.algo != 'fedsubmuon_gt':
            return {}
        if len(client_payloads) == 0:
            return {
                'gt_refresh_round': int(bool(is_refresh_round)),
                'gt_refresh_index': 0.0,
                'gt_topk': int(gt_topk),
                'gt_refresh_side': 0.0,
                'gt_update_mode_code': 0.0,
                'gt_basis_init_mode_code': 0.0,
                'gt_rank1_approx': float(int(gt_rank1_approx)),
                'gt_target_rel_step': float(gt_target_rel_step),
                'gt_rel_step_active': float(int(gt_target_rel_step > 0.0)),
                'gt_u_tangent_norm': 0.0,
                'gt_v_tangent_norm': 0.0,
                'gt_u_sigma_top': 0.0,
                'gt_v_sigma_top': 0.0,
                'gt_u_effective_step': 0.0,
                'gt_v_effective_step': 0.0,
                'gt_u_res_norm': 0.0,
                'gt_v_res_norm': 0.0,
                'gt_u_step_norm': 0.0,
                'gt_v_step_norm': 0.0,
                'gt_x_inherit_norm': 0.0,
                'gt_residual_norm': 0.0,
                'gt_basis_orth_err': 0.0,
                'gt_u_topk_active': 0.0,
                'gt_v_topk_active': 0.0,
                'gt_u_topk_score_sum': 0.0,
                'gt_v_topk_score_sum': 0.0,
            }

        weight_array = self._get_client_weight_array(selected_client_list)
        new_x = {name: torch.zeros_like(val) for name, val in self.x_global.items()}
        for client_idx, payload in enumerate(client_payloads):
            w = float(weight_array[client_idx])
            for name in new_x.keys():
                if name not in payload.get('x', {}):
                    continue
                new_x[name] += payload['x'][name].to(dtype=new_x[name].dtype) * w

        metrics = {
            'gt_refresh_round': int(bool(is_refresh_round)),
            'gt_refresh_index': 0.0,
            'gt_topk': int(gt_topk),
            'gt_refresh_side': 0.0,
            'gt_update_mode_code': 0.0,
            'gt_basis_init_mode_code': 0.0,
            'gt_rank1_approx': float(int(gt_rank1_approx)),
            'gt_target_rel_step': float(gt_target_rel_step),
            'gt_rel_step_active': float(int(gt_target_rel_step > 0.0)),
            'gt_u_tangent_norm': 0.0,
            'gt_v_tangent_norm': 0.0,
            'gt_u_sigma_top': 0.0,
            'gt_v_sigma_top': 0.0,
            'gt_u_effective_step': 0.0,
            'gt_v_effective_step': 0.0,
            'gt_u_res_norm': 0.0,
            'gt_v_res_norm': 0.0,
            'gt_u_step_norm': 0.0,
            'gt_v_step_norm': 0.0,
            'gt_x_inherit_norm': 0.0,
            'gt_residual_norm': 0.0,
            'gt_basis_orth_err': 0.0,
            'gt_u_topk_active': 0.0,
            'gt_v_topk_active': 0.0,
            'gt_u_topk_score_sum': 0.0,
            'gt_v_topk_score_sum': 0.0,
        }
        if not bool(is_refresh_round):
            self.x_global = new_x
            return metrics

        side_to_code = {'none': 0.0, 'left': 1.0, 'right': 2.0, 'both': 3.0}
        mode_to_code = {'both': 0.0, 'left': 1.0, 'right': 2.0, 'alternate_lr': 3.0, 'alternate_rl': 4.0}
        basis_init_to_code = {'random': 0.0, 'svd_left': 1.0, 'svd_right': 2.0, 'svd_both': 3.0}
        update_u, update_v, refresh_side, refresh_idx, update_mode = self._resolve_gt_refresh_side(cur_round)
        metrics['gt_refresh_index'] = float(refresh_idx)
        metrics['gt_refresh_side'] = float(side_to_code.get(refresh_side, 0.0))
        metrics['gt_update_mode_code'] = float(mode_to_code.get(update_mode, 0.0))
        metrics['gt_basis_init_mode_code'] = float(basis_init_to_code.get(self._resolve_gt_basis_init_mode(), 0.0))

        h_u_bar = {name: torch.zeros_like(self.u_global[name], dtype=torch.float32) for name in self.u_global.keys()}
        h_v_bar = {name: torch.zeros_like(self.v_basis_global[name], dtype=torch.float32) for name in self.v_basis_global.keys()}
        for client_idx, payload in enumerate(client_payloads):
            w = float(weight_array[client_idx])
            h_u_local = payload.get('h_u', {})
            h_v_local = payload.get('h_v', {})
            if not isinstance(h_u_local, dict):
                h_u_local = {}
            if not isinstance(h_v_local, dict):
                h_v_local = {}
            for layer_name in h_u_bar.keys():
                if layer_name in h_u_local and isinstance(h_u_local[layer_name], torch.Tensor):
                    h_u_bar[layer_name].add_(h_u_local[layer_name].to(dtype=torch.float32), alpha=w)
                if layer_name in h_v_local and isinstance(h_v_local[layer_name], torch.Tensor):
                    h_v_bar[layer_name].add_(h_v_local[layer_name].to(dtype=torch.float32), alpha=w)

        x_next = {}
        u_next = {}
        v_next = {}
        gt_sub_lr = float(getattr(self.args, 'gt_sub_lr', 0.1))
        merge_residual = bool(getattr(self.args, 'gt_merge_residual', False))

        for layer_name in new_x.keys():
            if layer_name not in self.u_global or layer_name not in self.v_basis_global:
                raise RuntimeError(f'[fedsubmuon_gt] missing persistent basis for layer={layer_name}')
            if layer_name not in self.seeds:
                raise RuntimeError(f'[fedsubmuon_gt] missing old seed for layer={layer_name} at refresh round')
            U_old = self.u_global[layer_name].float()
            V_old = self.v_basis_global[layer_name].float()
            X_agg = new_x[layer_name].float()
            expected_x_shape = (int(U_old.shape[1]), int(V_old.shape[1]))
            if tuple(X_agg.shape) != expected_x_shape:
                raise RuntimeError(
                    f'[fedsubmuon_gt] X shape mismatch @ {layer_name}: '
                    f'got={tuple(X_agg.shape)}, expected={expected_x_shape}'
                )

            H_u = h_u_bar[layer_name].float()
            H_v = h_v_bar[layer_name].float()
            if tuple(H_u.shape) != tuple(U_old.shape):
                raise RuntimeError(
                    f'[fedsubmuon_gt] H_u shape mismatch @ {layer_name}: '
                    f'got={tuple(H_u.shape)}, expected={tuple(U_old.shape)}'
                )
            if tuple(H_v.shape) != tuple(V_old.shape):
                raise RuntimeError(
                    f'[fedsubmuon_gt] H_v shape mismatch @ {layer_name}: '
                    f'got={tuple(H_v.shape)}, expected={tuple(V_old.shape)}'
                )

            B_u = U_old.t() @ H_u
            R_u = H_u - U_old @ B_u
            u_step = R_u @ B_u.t()
            if update_u:
                u_scores = torch.linalg.norm(u_step, dim=0)
                if gt_topk > 0:
                    k_eff_u = min(int(gt_topk), int(u_step.shape[1]))
                    if k_eff_u > 0:
                        idx_u = torch.topk(u_scores, k=k_eff_u, largest=True).indices
                        u_step_applied = torch.zeros_like(u_step)
                        u_step_applied[:, idx_u] = u_step[:, idx_u]
                        metrics['gt_u_topk_score_sum'] += float(u_scores[idx_u].sum().item())
                    else:
                        idx_u = None
                        u_step_applied = torch.zeros_like(u_step)
                else:
                    k_eff_u = int(u_step.shape[1])
                    idx_u = None
                    u_step_applied = u_step
                    metrics['gt_u_topk_score_sum'] += float(u_scores.sum().item())

                if gt_rank1_approx:
                    u_tangent_use, u_sigma_top = self._rank1_approx_tangent(u_step_applied)
                else:
                    u_tangent_use = u_step_applied
                    u_sigma_top = 0.0
                eta_u = self._resolve_gt_effective_step(
                    basis=U_old,
                    tangent_used=u_tangent_use,
                    sigma_top=u_sigma_top,
                    rank1_enabled=gt_rank1_approx,
                    gt_sub_lr=gt_sub_lr,
                    gt_target_rel_step=gt_target_rel_step,
                )

                U_new_raw = U_old + eta_u * u_tangent_use
                U_new = self._orthonormalize_columns(U_new_raw)
            else:
                k_eff_u = 0
                idx_u = None
                u_step_applied = torch.zeros_like(u_step)
                u_tangent_use = torch.zeros_like(u_step)
                u_sigma_top = 0.0
                eta_u = 0.0
                U_new = U_old.clone()
            metrics['gt_u_topk_active'] += float(k_eff_u)

            B_v = V_old.t() @ H_v
            R_v = H_v - V_old @ B_v
            v_step = R_v @ B_v.t()
            if update_v:
                v_scores = torch.linalg.norm(v_step, dim=0)
                if gt_topk > 0:
                    k_eff_v = min(int(gt_topk), int(v_step.shape[1]))
                    if k_eff_v > 0:
                        idx_v = torch.topk(v_scores, k=k_eff_v, largest=True).indices
                        v_step_applied = torch.zeros_like(v_step)
                        v_step_applied[:, idx_v] = v_step[:, idx_v]
                        metrics['gt_v_topk_score_sum'] += float(v_scores[idx_v].sum().item())
                    else:
                        idx_v = None
                        v_step_applied = torch.zeros_like(v_step)
                else:
                    k_eff_v = int(v_step.shape[1])
                    idx_v = None
                    v_step_applied = v_step
                    metrics['gt_v_topk_score_sum'] += float(v_scores.sum().item())

                if gt_rank1_approx:
                    v_tangent_use, v_sigma_top = self._rank1_approx_tangent(v_step_applied)
                else:
                    v_tangent_use = v_step_applied
                    v_sigma_top = 0.0
                eta_v = self._resolve_gt_effective_step(
                    basis=V_old,
                    tangent_used=v_tangent_use,
                    sigma_top=v_sigma_top,
                    rank1_enabled=gt_rank1_approx,
                    gt_sub_lr=gt_sub_lr,
                    gt_target_rel_step=gt_target_rel_step,
                )

                V_new_raw = V_old + eta_v * v_tangent_use
                V_new = self._orthonormalize_columns(V_new_raw)
            else:
                k_eff_v = 0
                idx_v = None
                v_step_applied = torch.zeros_like(v_step)
                v_tangent_use = torch.zeros_like(v_step)
                v_sigma_top = 0.0
                eta_v = 0.0
                V_new = V_old.clone()
            metrics['gt_v_topk_active'] += float(k_eff_v)

            delta_old = U_old @ X_agg @ V_old.t()
            layer_shape = self.submuon_layer_dims.get(layer_name, (int(U_old.shape[0]), int(V_old.shape[0])))
            if tuple(delta_old.shape) != tuple(layer_shape):
                raise RuntimeError(
                    f'[fedsubmuon_gt] UXV^T shape mismatch @ {layer_name}: '
                    f'got={tuple(delta_old.shape)}, expected={tuple(layer_shape)}'
                )

            X_new = U_new.t() @ delta_old @ V_new
            residual = delta_old - (U_new @ X_new @ V_new.t())
            if merge_residual:
                self._merge_layer_residual_to_backbone(layer_name, residual)

            x_next[layer_name] = X_new.cpu().contiguous()
            u_next[layer_name] = U_new.cpu().contiguous()
            v_next[layer_name] = V_new.cpu().contiguous()

            eye_u = torch.eye(U_new.shape[1], dtype=torch.float32, device=U_new.device)
            eye_v = torch.eye(V_new.shape[1], dtype=torch.float32, device=V_new.device)
            u_orth_err = torch.linalg.norm(U_new.t() @ U_new - eye_u)
            v_orth_err = torch.linalg.norm(V_new.t() @ V_new - eye_v)

            metrics['gt_u_res_norm'] += float(torch.linalg.norm(R_u).item())
            metrics['gt_v_res_norm'] += float(torch.linalg.norm(R_v).item())
            metrics['gt_u_step_norm'] += float(torch.linalg.norm(u_tangent_use).item())
            metrics['gt_v_step_norm'] += float(torch.linalg.norm(v_tangent_use).item())
            metrics['gt_u_tangent_norm'] += float(torch.linalg.norm(u_tangent_use).item())
            metrics['gt_v_tangent_norm'] += float(torch.linalg.norm(v_tangent_use).item())
            metrics['gt_u_sigma_top'] += float(u_sigma_top)
            metrics['gt_v_sigma_top'] += float(v_sigma_top)
            metrics['gt_u_effective_step'] += float(eta_u)
            metrics['gt_v_effective_step'] += float(eta_v)
            metrics['gt_x_inherit_norm'] += float(torch.linalg.norm(X_new).item())
            metrics['gt_residual_norm'] += float(torch.linalg.norm(residual).item())
            metrics['gt_basis_orth_err'] = max(
                float(metrics['gt_basis_orth_err']),
                float(u_orth_err.item()),
                float(v_orth_err.item()),
            )

        max_x_abs = 0.0
        for tensor in x_next.values():
            if tensor.numel() > 0:
                max_x_abs = max(max_x_abs, float(torch.max(torch.abs(tensor)).item()))
        if not np.isfinite(max_x_abs):
            raise RuntimeError('[fedsubmuon_gt] inherited X contains non-finite values after refresh')

        old_seeds = dict(self.seeds)
        new_seeds = {}
        for layer_name in self.seeds.keys():
            new_seeds[layer_name] = int(self.seed_rng.randint(1, 2**31 - 1))

        self.x_global = x_next
        self.u_global = u_next
        self.v_basis_global = v_next
        self.seeds = new_seeds

        if bool(getattr(self.args, 'log', False)):
            sample_layer = next(iter(self.seeds.keys())) if len(self.seeds) > 0 else 'n/a'
            old_seed = old_seeds.get(sample_layer, None) if sample_layer != 'n/a' else None
            new_seed = self.seeds.get(sample_layer, None) if sample_layer != 'n/a' else None
            print(
                f'[fedsubmuon_gt][server] refresh_round={int(cur_round)} old_seed={old_seed} new_seed={new_seed} '
                f'basis_init_mode={self._resolve_gt_basis_init_mode()} '
                f'gt_update_mode={update_mode} refresh_index={int(refresh_idx)} refresh_side={refresh_side} '
                f'gt_topk={int(gt_topk)} '
                f'gt_rank1={int(gt_rank1_approx)} gt_target_rel_step={float(gt_target_rel_step):.6e} '
                f'u_eta={metrics["gt_u_effective_step"]:.6e} v_eta={metrics["gt_v_effective_step"]:.6e} '
                f'u_res={metrics["gt_u_res_norm"]:.6e} v_res={metrics["gt_v_res_norm"]:.6e} '
                f'u_topk_active={int(metrics["gt_u_topk_active"])} '
                f'v_topk_active={int(metrics["gt_v_topk_active"])} '
                f'x_inherit={metrics["gt_x_inherit_norm"]:.6e} residual={metrics["gt_residual_norm"]:.6e}'
            )
        return metrics

    def trigger_rebase(self, cur_round):
        return self.maybe_refresh_submuon_seeds(cur_round=cur_round, force=True)

    def aggregate_submuon(self, client_payloads, selected_client_list):
        weight_array = self._get_client_weight_array(selected_client_list)
        aggregate_muon_state = bool(self.algo == 'fedsubmuon' and getattr(self.args, 'aggregate_muon_state', False))

        new_x = {name: torch.zeros_like(val) for name, val in self.x_global.items()}
        new_m = {name: torch.zeros_like(val) for name, val in self.m_global.items()} if aggregate_muon_state else None

        for client_idx, payload in enumerate(client_payloads):
            w = float(weight_array[client_idx])
            for name in new_x.keys():
                new_x[name] += payload['x'][name].to(dtype=new_x[name].dtype) * w
                if aggregate_muon_state:
                    new_m[name] += payload['m'][name].to(dtype=new_m[name].dtype) * w

        self.x_global = new_x
        if aggregate_muon_state:
            self.m_global = new_m

    def aggregate_fedmultisubmuon(self, client_payloads, selected_client_list, cur_round=1):
        if self.algo != 'fedmultisubmuon' or len(client_payloads) == 0:
            return
        weight_array = self._get_client_weight_array(selected_client_list)
        active_keys = list(self.global_multisub_selected_keys)
        if len(active_keys) == 0:
            active_keys = list(self.global_multisub_b_state.keys())

        for sub_key in active_keys:
            if sub_key not in self.global_multisub_b_state or sub_key not in self.global_multisub_c_state:
                continue
            accum_b = torch.zeros_like(self.global_multisub_b_state[sub_key], dtype=torch.float32)
            accum_c = torch.zeros_like(self.global_multisub_c_state[sub_key], dtype=torch.float32)
            score_acc = 0.0
            score_w = 0.0
            b_w = 0.0
            c_w = 0.0
            for client_idx, payload in enumerate(client_payloads):
                weight = float(weight_array[client_idx])
                b_local = payload.get('b', {})
                c_local = payload.get('c', {})
                if isinstance(b_local, dict) and sub_key in b_local:
                    b_tensor = b_local[sub_key]
                    if isinstance(b_tensor, torch.Tensor):
                        accum_b.add_(b_tensor.to(dtype=torch.float32), alpha=weight)
                        b_w += weight
                if isinstance(c_local, dict) and sub_key in c_local:
                    c_tensor = c_local[sub_key]
                    if isinstance(c_tensor, torch.Tensor):
                        accum_c.add_(c_tensor.to(dtype=torch.float32), alpha=weight)
                        c_w += weight

                score_local = payload.get('scores', {})
                if isinstance(score_local, dict) and sub_key in score_local:
                    score_val = float(score_local[sub_key])
                    if np.isfinite(score_val):
                        score_acc += score_val * weight
                        score_w += weight

            if b_w > 0.0:
                self.global_multisub_b_state[sub_key] = (accum_b / b_w).cpu().contiguous()
            if c_w > 0.0:
                self.global_multisub_c_state[sub_key] = (accum_c / c_w).cpu().contiguous()
            if score_w > 0.0:
                self.global_multisub_scores[sub_key] = float(score_acc / score_w)

        self.global_multisub_selected_keys = select_topk_subspaces(
            self.global_multisub_scores,
            int(getattr(self.args, 'multisub_topk', 0)),
        )
        if len(self.global_multisub_selected_keys) == 0:
            self.global_multisub_selected_keys = list(self.global_multisub_b_state.keys())
        if (not self._multisub_debug_logged) and int(cur_round) >= 1:
            top_items = sorted(
                self.global_multisub_scores.items(),
                key=lambda kv: (-float(kv[1]), kv[0]),
            )[:3]
            print(f'[debug][fedmultisubmuon][server] round={cur_round} top_scores={top_items}')
            self._multisub_debug_logged = True

    def aggregate_fedstructmuon(self, client_payloads, selected_client_list, cur_round=1):
        if self.algo != 'fedstructmuon' or len(client_payloads) == 0:
            return
        weight_array = self._get_client_weight_array(selected_client_list)
        active_keys = list(self.global_struct_selected_keys)
        if len(active_keys) == 0:
            active_keys = list(self.global_struct_x_state.keys())

        for sub_key in active_keys:
            if sub_key not in self.global_struct_x_state:
                continue
            accum_x = torch.zeros_like(self.global_struct_x_state[sub_key], dtype=torch.float32)
            score_acc = 0.0
            x_w = 0.0
            score_w = 0.0
            for client_idx, payload in enumerate(client_payloads):
                weight = float(weight_array[client_idx])
                x_local = payload.get('x', {})
                if isinstance(x_local, dict) and sub_key in x_local:
                    x_tensor = x_local[sub_key]
                    if isinstance(x_tensor, torch.Tensor):
                        if tuple(x_tensor.shape) != tuple(self.global_struct_x_state[sub_key].shape):
                            raise RuntimeError(
                                f'[fedstructmuon] X shape mismatch in aggregation for {sub_key}: '
                                f'global={tuple(self.global_struct_x_state[sub_key].shape)}, '
                                f'client={tuple(x_tensor.shape)}'
                            )
                        accum_x.add_(x_tensor.to(dtype=torch.float32), alpha=weight)
                        x_w += weight
                score_local = payload.get('scores', {})
                if isinstance(score_local, dict) and sub_key in score_local:
                    score_val = float(score_local[sub_key])
                    if np.isfinite(score_val):
                        score_acc += score_val * weight
                        score_w += weight

            if x_w > 0.0:
                self.global_struct_x_state[sub_key] = (accum_x / x_w).cpu().contiguous()
            if score_w > 0.0:
                self.global_struct_scores[sub_key] = float(score_acc / score_w)

        struct_topk = self._get_scheduled_struct_topk(cur_round)
        self.global_struct_selected_keys = select_topk_subspaces(
            self.global_struct_scores,
            struct_topk,
        )
        if len(self.global_struct_selected_keys) == 0:
            self.global_struct_selected_keys = list(self.global_struct_x_state.keys())
        if (not self._struct_debug_logged) and int(cur_round) >= 1:
            top_items = sorted(
                self.global_struct_scores.items(),
                key=lambda kv: (-float(kv[1]), kv[0]),
            )[:3]
            print(
                f'[debug][fedstructmuon][server] round={cur_round} '
                f'scheduled_topk={struct_topk}, top_scores={top_items}'
            )
            self._struct_debug_logged = True

    def _get_scheduled_struct_topk(self, cur_round):
        total_kk = int(len(self.global_struct_scores))
        if total_kk <= 0:
            return 0

        target_kk = int(getattr(self.args, 'struct_topk', 0))
        if target_kk <= 0 or target_kk >= total_kk:
            return total_kk

        initial_warmup = max(int(getattr(self.args, 'struct_topk_init_warmup', 1)), 0)
        final_warmup = int(getattr(self.args, 'struct_topk_final_warmup', int(getattr(self.args, 'rounds', 1))))
        tt = max(float(getattr(self.args, 'struct_topk_tt', 1.0)), 1e-12)
        step = int(cur_round)

        if final_warmup <= initial_warmup:
            curr_kk = target_kk
        elif step < initial_warmup:
            curr_kk = total_kk
        elif step > final_warmup:
            curr_kk = target_kk
        else:
            mul_coeff = 1.0 - float(step - initial_warmup) / float(final_warmup - initial_warmup)
            mul_coeff = min(max(mul_coeff, 0.0), 1.0)
            curr_kk = int(target_kk + (total_kk - target_kk) * (mul_coeff ** tt))

        curr_kk = int(max(target_kk, min(total_kk, curr_kk)))
        if not self._struct_topk_schedule_logged:
            print(
                f'[fedstructmuon] AdaMSS top-k schedule: '
                f'total={total_kk}, target={target_kk}, '
                f'init_warmup={initial_warmup}, final_warmup={final_warmup}, tt={tt}'
            )
            self._struct_topk_schedule_logged = True
        return curr_kk

    def aggregate_lora(self, client_payloads, selected_client_list):
        if len(client_payloads) == 0:
            return

        weight_array = self._get_client_weight_array(selected_client_list)

        if self.algo in ['fedit', 'federa']:
            state_keys = list(client_payloads[0]['lora_state'].keys())
            new_global_lora = {key: torch.zeros_like(client_payloads[0]['lora_state'][key], dtype=torch.float32) for key in state_keys}

            for client_idx, payload in enumerate(client_payloads):
                w = float(weight_array[client_idx])
                local_state = payload['lora_state']
                for key in state_keys:
                    new_global_lora[key] += local_state[key].to(dtype=torch.float32) * w
            self.global_lora_state = {k: v.cpu() for k, v in new_global_lora.items()}
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

    def aggregate_fedsalora(self, client_a_states, selected_client_list):
        if len(client_a_states) == 0:
            return

        weight_array = self._get_client_weight_array(selected_client_list)
        state_keys = list(client_a_states[0].keys())
        new_global_lora_a = {
            key: torch.zeros_like(client_a_states[0][key], dtype=torch.float32)
            for key in state_keys
        }
        for client_idx, local_a_state in enumerate(client_a_states):
            weight = float(weight_array[client_idx])
            for key in state_keys:
                if key not in local_a_state:
                    continue
                new_global_lora_a[key] += local_a_state[key].to(dtype=torch.float32) * weight
        self.global_lora_A_state = {k: v.cpu() for k, v in new_global_lora_a.items()}

    def aggregate_fedexlora(self, client_payloads, selected_client_list, cur_round=1):
        if len(client_payloads) == 0:
            return

        weight_array = self._get_client_weight_array(selected_client_list)

        # 1) classifier weighted average
        classifier_keys = sorted(
            {
                key
                for payload in client_payloads
                for key in payload.get('classifier_state', {}).keys()
            }
        )
        new_global_classifier = {}
        for key in classifier_keys:
            ref_tensor = None
            for payload in client_payloads:
                local_classifier = payload.get('classifier_state', {})
                if key in local_classifier:
                    ref_tensor = local_classifier[key]
                    break
            if ref_tensor is None:
                continue
            avg_tensor = torch.zeros_like(ref_tensor, dtype=torch.float32)
            for client_idx, payload in enumerate(client_payloads):
                local_classifier = payload.get('classifier_state', {})
                if key not in local_classifier:
                    continue
                avg_tensor += local_classifier[key].to(dtype=torch.float32) * float(weight_array[client_idx])
            new_global_classifier[key] = avg_tensor.cpu()
        self.global_classifier_state = new_global_classifier
        if len(self.global_classifier_state) > 0:
            load_classifier_state(self.model, self.global_classifier_state)

        # 2) FedEx-LoRA: average A/B and compensate residual into base weight.
        first_a_state = client_payloads[0].get('lora_A_state', {})
        first_b_state = client_payloads[0].get('lora_B_state', {})
        first_lora_state = {}
        first_lora_state.update(first_a_state)
        first_lora_state.update(first_b_state)
        layer_pairs = get_lora_pair_keys(first_lora_state)
        name_to_module = dict(self.model.named_modules())
        new_global_lora_a = {}
        new_global_lora_b = {}

        for layer_name, pair in layer_pairs.items():
            if 'a_key' not in pair or 'b_key' not in pair:
                continue
            a_key = pair['a_key']
            b_key = pair['b_key']

            resolved_layer = resolve_layer_name_for_model(layer_name, self.model)
            module = name_to_module.get(resolved_layer, None)
            if module is None or (not hasattr(module, 'weight')):
                raise RuntimeError(f'[fedexlora] cannot resolve base layer for {layer_name} -> {resolved_layer}')

            base_weight = module.weight.detach().cpu()
            if base_weight.ndim != 2:
                raise RuntimeError(
                    f'[fedexlora] expected 2D base weight at {resolved_layer}, got shape={tuple(base_weight.shape)}'
                )
            out_dim, in_dim = int(base_weight.shape[0]), int(base_weight.shape[1])
            rank = max(int(getattr(self.args, 'lora_r', 0)), 1)

            A_bar = torch.zeros_like(first_a_state[a_key], dtype=torch.float32)
            B_bar = torch.zeros_like(first_b_state[b_key], dtype=torch.float32)
            M = torch.zeros((out_dim, in_dim), dtype=torch.float32)

            for client_idx, payload in enumerate(client_payloads):
                local_a_state = payload.get('lora_A_state', {})
                local_b_state = payload.get('lora_B_state', {})
                if a_key not in local_a_state or b_key not in local_b_state:
                    raise RuntimeError(f'[fedexlora] missing LoRA keys for layer={layer_name}, a={a_key}, b={b_key}')

                a_i = local_a_state[a_key].to(dtype=torch.float32)
                b_i = local_b_state[b_key].to(dtype=torch.float32)

                if a_i.ndim != 2 or b_i.ndim != 2:
                    raise RuntimeError(
                        f'[fedexlora] invalid tensor rank at layer={layer_name}: '
                        f'A.shape={tuple(a_i.shape)}, B.shape={tuple(b_i.shape)}'
                    )
                if b_i.shape[1] != a_i.shape[0]:
                    raise RuntimeError(
                        f'[fedexlora] matmul mismatch at layer={layer_name}: '
                        f'A.shape={tuple(a_i.shape)}, B.shape={tuple(b_i.shape)}'
                    )
                if b_i.shape != (out_dim, rank) or a_i.shape != (rank, in_dim):
                    raise RuntimeError(
                        f'[fedexlora] shape mismatch at layer={layer_name} ({resolved_layer}): '
                        f'expected B={(out_dim, rank)}, A={(rank, in_dim)}, got B={tuple(b_i.shape)}, A={tuple(a_i.shape)}'
                    )

                local_prod = torch.matmul(b_i, a_i)
                if tuple(local_prod.shape) != (out_dim, in_dim):
                    raise RuntimeError(
                        f'[fedexlora] product shape mismatch at layer={layer_name}: '
                        f'(B@A).shape={tuple(local_prod.shape)} vs base_weight.shape={(out_dim, in_dim)}'
                    )

                weight = float(weight_array[client_idx])
                M += local_prod * weight
                A_bar += a_i * weight
                B_bar += b_i * weight

            BA_bar = torch.matmul(B_bar, A_bar)
            if tuple(BA_bar.shape) != (out_dim, in_dim):
                raise RuntimeError(
                    f'[fedexlora] (B_bar@A_bar) shape mismatch at layer={layer_name}: '
                    f'got {tuple(BA_bar.shape)}, expected {(out_dim, in_dim)}'
                )
            residual = M - BA_bar
            scaled_residual = residual * float(self.flora_scaling)
            with torch.no_grad():
                module.weight.data.add_(
                    scaled_residual.to(device=module.weight.device, dtype=module.weight.dtype)
                )

            new_global_lora_a[a_key] = A_bar.cpu()
            new_global_lora_b[b_key] = B_bar.cpu()

            if (not self._fedex_debug_logged) and int(cur_round) == 1:
                print(
                    f'[debug][fedexlora][server] layer={resolved_layer} '
                    f'||M||={float(torch.linalg.norm(M).item()):.6e}, '
                    f'||BbarAbar||={float(torch.linalg.norm(BA_bar).item()):.6e}, '
                    f'||R||={float(torch.linalg.norm(residual).item()):.6e}, '
                    f'||R*s||={float(torch.linalg.norm(scaled_residual).item()):.6e}'
                )
                self._fedex_debug_logged = True

        self.global_lora_A_state = new_global_lora_a
        self.global_lora_B_state = new_global_lora_b

    def aggregate_florg(self, client_payloads, selected_client_list, cur_round=1):
        if len(client_payloads) == 0:
            return
        if len(self.global_florg_A_state) == 0:
            return

        weight_array = self._get_client_weight_array(selected_client_list)
        new_global_florg_a = {}

        for key, prev_a in self.global_florg_A_state.items():
            prev_a_f32 = prev_a.to(dtype=torch.float32)
            if prev_a_f32.ndim != 2:
                raise RuntimeError(f'[florg] expected 2D A tensor for key={key}, got shape={tuple(prev_a_f32.shape)}')
            rank_r, k = int(prev_a_f32.shape[0]), int(prev_a_f32.shape[1])
            gram_q = torch.zeros((k, k), dtype=torch.float32)

            for client_idx, payload in enumerate(client_payloads):
                local_state = payload.get('florg_A', {})
                if key not in local_state:
                    raise RuntimeError(f'[florg] missing key in client payload: {key}')
                local_a = local_state[key].to(dtype=torch.float32)
                if tuple(local_a.shape) != (rank_r, k):
                    raise RuntimeError(
                        f'[florg] shape mismatch for {key}: expected {(rank_r, k)}, got {tuple(local_a.shape)}'
                    )
                weight = float(weight_array[client_idx])
                gram_q += torch.matmul(local_a.t(), local_a) * weight

            gram_q = 0.5 * (gram_q + gram_q.t())
            eigvals, eigvecs = torch.linalg.eigh(gram_q)
            sort_idx = torch.argsort(eigvals, descending=True)
            top_idx = sort_idx[:rank_r]
            top_vals = eigvals[top_idx].clamp_min(0.0)
            top_vecs = eigvecs[:, top_idx]
            a_tilde = torch.matmul(torch.diag(torch.sqrt(top_vals)), top_vecs.t())

            procrustes_m = torch.matmul(prev_a_f32, a_tilde.t())
            U, _, Vh = torch.linalg.svd(procrustes_m, full_matrices=False)
            S = torch.matmul(U, Vh)
            a_next = torch.matmul(S, a_tilde)
            new_global_florg_a[key] = a_next.cpu().contiguous()

            if (not self._florg_debug_logged) and int(cur_round) >= 1:
                top3 = torch.sort(eigvals, descending=True).values[:3]
                approx_err = torch.linalg.norm(torch.matmul(a_next.t(), a_next) - gram_q)
                print(
                    f'[debug][florg][server] key={key} '
                    f'top_eigs={[float(v.item()) for v in top3]} '
                    f'||Q||={float(torch.linalg.norm(gram_q).item()):.6e} '
                    f'||A_next^T A_next - Q||={float(approx_err.item()):.6e}'
                )
                self._florg_debug_logged = True

        self.global_florg_A_state = new_global_florg_a

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
        self.global_named_optim_state = {'state': {}}
        self.fedavg_weight_array = None
        self.fedavg_accumulator = {}
        self.fedavg_non_float_state = {}
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
        if self.algo not in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd'] or not self.args.save:
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
                'm_global': {k: v.cpu() for k, v in self.m_global.items()} if self.algo == 'fedsubmuon' else None,
                'v_global': None,
                'seeds': dict(self.seeds),
                'u_global': {k: v.cpu() for k, v in self.u_global.items()} if self.algo == 'fedsubmuon_gt' else None,
                'v_basis_global': {k: v.cpu() for k, v in self.v_basis_global.items()} if self.algo == 'fedsubmuon_gt' else None,
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
                    'stop_F': int(getattr(self.args, 'stop_F', -1)),
                    'aggregate_muon_state': bool(getattr(self.args, 'aggregate_muon_state', False)),
                    'gt_probe_batches': int(getattr(self.args, 'gt_probe_batches', 1)),
                    'gt_sub_lr': float(getattr(self.args, 'gt_sub_lr', 0.1)),
                    'gt_topk': int(getattr(self.args, 'gt_topk', 0)),
                    'gt_merge_residual': bool(getattr(self.args, 'gt_merge_residual', False)),
                    'gt_rank1_approx': bool(getattr(self.args, 'gt_rank1_approx', False)),
                    'gt_target_rel_step': float(getattr(self.args, 'gt_target_rel_step', 0.0)),
                    'basis_init_mode': str(getattr(self.args, 'basis_init_mode', 'random')),
                    'gt_update_mode': str(getattr(self.args, 'gt_update_mode', 'both')),
                    'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                },
            },
            ckpt_path,
        )
        print(f'[ckpt] saved to: {ckpt_path}')
        return True

    def save_best_multisub_ckpt(self, metric, cur_round):
        if self.algo != 'fedmultisubmuon' or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if not improved:
            return False

        self.best_metric = metric
        metadata_cpu = {}
        for key, meta in self.global_multisub_metadata.items():
            metadata_cpu[key] = {
                'layer_name': str(meta['layer_name']),
                'indices': meta['indices'].cpu(),
                'A': meta['A'].cpu(),
                'rank_big': int(meta.get('rank_big', int(self.global_multisub_b_state[key].shape[0]))),
                'rank_small': int(meta.get('rank_small', int(self.global_multisub_c_state[key].shape[0]))),
                'flat_id': int(meta.get('flat_id', -1)),
            }

        ckpt_path = os.path.join(self._get_ckpt_dir(), 'best.pt')
        torch.save(
            {
                'algo': 'fedmultisubmuon',
                'backbone_state_dict': self.model.state_dict(),
                'global_multisub_b_state': {k: v.cpu() for k, v in self.global_multisub_b_state.items()},
                'global_multisub_c_state': {k: v.cpu() for k, v in self.global_multisub_c_state.items()},
                'global_multisub_metadata': metadata_cpu,
                'global_multisub_scores': {k: float(v) for k, v in self.global_multisub_scores.items()},
                'global_multisub_selected_keys': list(self.global_multisub_selected_keys),
                'round': int(cur_round),
                'best_metric': float(metric),
                'hparams': {
                    'algo': self.args.algo,
                    'rank_r': int(self.args.rank_r),
                    'svd_rank': int(getattr(self.args, 'svd_rank', 500)),
                    'beta': float(getattr(self.args, 'beta', 0.95)),
                    'ns_steps': int(getattr(self.args, 'ns_steps', 5)),
                    'lr': float(self.args.lr),
                    'multisub_num_subspaces': int(getattr(self.args, 'multisub_num_subspaces', 4)),
                    'multisub_topk': int(getattr(self.args, 'multisub_topk', 0)),
                    'multisub_score_interval': int(getattr(self.args, 'multisub_score_interval', 1)),
                    'multisub_score_beta1': float(getattr(self.args, 'multisub_score_beta1', 0.9)),
                    'multisub_score_beta2': float(getattr(self.args, 'multisub_score_beta2', 0.999)),
                    'multisub_seed_base': int(getattr(self.args, 'multisub_seed_base', int(self.args.seed) + 13579)),
                    'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                },
            },
            ckpt_path,
        )
        print(f'[ckpt] saved to: {ckpt_path}')
        return True

    def save_best_struct_ckpt(self, metric, cur_round):
        if self.algo != 'fedstructmuon' or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if not improved:
            return False

        self.best_metric = metric
        metadata_cpu = {}
        for key, meta in self.global_struct_metadata.items():
            metadata_cpu[key] = {
                'layer_name': str(meta['layer_name']),
                'indices': meta['indices'].cpu(),
                'A': meta['A'].cpu(),
                'V': meta['V'].cpu(),
                'rank': int(meta.get('rank', int(self.global_struct_x_state[key].shape[0]))),
                'rank_left': int(meta.get('rank_left', int(self.global_struct_x_state[key].shape[0]))),
                'rank_right': int(meta.get('rank_right', int(self.global_struct_x_state[key].shape[1]))),
                'seed': int(meta.get('seed', -1)),
                'flat_id': int(meta.get('flat_id', -1)),
            }

        ckpt_path = os.path.join(self._get_ckpt_dir(), 'best.pt')
        torch.save(
            {
                'algo': 'fedstructmuon',
                'backbone_state_dict': self.model.state_dict(),
                'global_struct_x_state': {k: v.cpu() for k, v in self.global_struct_x_state.items()},
                'global_struct_metadata': metadata_cpu,
                'global_struct_scores': {k: float(v) for k, v in self.global_struct_scores.items()},
                'global_struct_selected_keys': list(self.global_struct_selected_keys),
                'round': int(cur_round),
                'best_metric': float(metric),
                'hparams': {
                    'algo': self.args.algo,
                    'rank_r': int(self.args.rank_r),
                    'rank_left': int(getattr(self.args, 'rank_left', int(self.args.rank_r))),
                    'rank_right': int(getattr(self.args, 'rank_right', int(self.args.rank_r))),
                    'svd_rank': int(getattr(self.args, 'svd_rank', 500)),
                    'beta': float(getattr(self.args, 'beta', 0.95)),
                    'ns_steps': int(getattr(self.args, 'ns_steps', 5)),
                    'lr': float(self.args.lr),
                    'struct_num_subspaces': int(getattr(self.args, 'struct_num_subspaces', 4)),
                    'struct_topk': int(getattr(self.args, 'struct_topk', 0)),
                    'struct_score_interval': int(getattr(self.args, 'struct_score_interval', 10)),
                    'struct_seed_base': int(getattr(self.args, 'struct_seed_base', int(self.args.seed) + 24680)),
                    'struct_topk_init_warmup': int(getattr(self.args, 'struct_topk_init_warmup', 1)),
                    'struct_topk_final_warmup': int(getattr(self.args, 'struct_topk_final_warmup', int(getattr(self.args, 'rounds', 1)))),
                    'struct_topk_tt': float(getattr(self.args, 'struct_topk_tt', 1.0)),
                    'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                },
            },
            ckpt_path,
        )
        print(f'[ckpt] saved to: {ckpt_path}')
        return True

    def save_best_lora_ckpt(self, metric, cur_round):
        if self.algo not in ['fedit', 'federa', 'flora', 'fedsalora', 'fedexlora', 'florg'] or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if not improved:
            return False

        self.best_metric = metric

        ckpt_dir = self._get_ckpt_dir()
        ckpt_payload = {
            'algo': self.algo,
            'backbone_state_dict': self.model.state_dict(),
            'round': int(cur_round),
            'best_metric': float(self.best_metric),
            'metric': float(metric),
        }

        if self.algo in ['fedit', 'federa']:
            ckpt_payload['lora_hparams'] = {
                'lora_r': int(getattr(self.args, 'lora_r', 16)),
                'lora_alpha': float(getattr(self.args, 'lora_alpha', 16.0)),
                'lora_dropout': float(getattr(self.args, 'lora_dropout', 0.0)),
                'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                'lora_bias': getattr(self.args, 'lora_bias', 'none'),
                'scaling': float(self.flora_scaling),
                'federa_svd_dtype': str(getattr(self.args, 'federa_svd_dtype', 'fp32')),
            }
            ckpt_payload['global_lora_state'] = {k: v.cpu() for k, v in self.global_lora_state.items()}
        elif self.algo == 'flora':
            ckpt_payload['lora_hparams'] = {
                'lora_r': int(getattr(self.args, 'lora_r', 16)),
                'lora_alpha': float(getattr(self.args, 'lora_alpha', 16.0)),
                'lora_dropout': float(getattr(self.args, 'lora_dropout', 0.0)),
                'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                'lora_bias': getattr(self.args, 'lora_bias', 'none'),
                'scaling': float(self.flora_scaling),
            }
            ckpt_payload['global_deltaW_state'] = {k: v.cpu() for k, v in self.global_deltaW_state.items()}
        elif self.algo == 'fedsalora':
            ckpt_payload['lora_hparams'] = {
                'lora_r': int(getattr(self.args, 'lora_r', 16)),
                'lora_alpha': float(getattr(self.args, 'lora_alpha', 16.0)),
                'lora_dropout': float(getattr(self.args, 'lora_dropout', 0.0)),
                'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                'lora_bias': getattr(self.args, 'lora_bias', 'none'),
                'scaling': float(self.flora_scaling),
            }
            ckpt_payload['global_lora_A_state'] = {k: v.cpu() for k, v in self.global_lora_A_state.items()}
        elif self.algo == 'fedexlora':
            ckpt_payload['lora_hparams'] = {
                'lora_r': int(getattr(self.args, 'lora_r', 16)),
                'lora_alpha': float(getattr(self.args, 'lora_alpha', 16.0)),
                'lora_dropout': float(getattr(self.args, 'lora_dropout', 0.0)),
                'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
                'lora_bias': getattr(self.args, 'lora_bias', 'none'),
                'scaling': float(self.flora_scaling),
            }
            if len(self.global_lora_A_state) == 0 or len(self.global_lora_B_state) == 0:
                print(
                    '[warn][fedexlora] saving best checkpoint with empty LoRA state: '
                    f'|A|={len(self.global_lora_A_state)}, |B|={len(self.global_lora_B_state)}'
                )
            ckpt_payload['global_lora_A_state'] = {k: v.cpu() for k, v in self.global_lora_A_state.items()}
            ckpt_payload['global_lora_B_state'] = {k: v.cpu() for k, v in self.global_lora_B_state.items()}
            ckpt_payload['global_classifier_state'] = {k: v.cpu() for k, v in self.global_classifier_state.items()}
        else:
            if len(self.global_florg_A_state) == 0:
                print('[warn][florg] saving best checkpoint with empty florg A state')
            ckpt_payload['global_florg_A_state'] = {k: v.cpu() for k, v in self.global_florg_A_state.items()}
            ckpt_payload['global_florg_seed_state'] = self.get_florg_seed_state()
            ckpt_payload['global_florg_basis_state'] = self.get_florg_basis_state()
            ckpt_payload['florg_hparams'] = {
                'florg_rank_r': int(getattr(self.args, 'florg_rank_r', 16)),
                'florg_seed_base': int(getattr(self.args, 'florg_seed_base', 95317)),
                'lora_target_modules': getattr(self.args, 'lora_target_modules', None),
            }

        best_ckpt_path = os.path.join(ckpt_dir, 'best.pt')
        torch.save(ckpt_payload, best_ckpt_path)
        print(f'[ckpt] saved to: {best_ckpt_path}')
        return True

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
        return eval_metric

    def eval_loss(self, cur_round):
        eval_model = None
        framework = None
        temp_eval_model = False
        if self.algo in ['fedit', 'federa']:
            eval_model = build_lora_model(deepcopy(self.model), self.args).to(self.device)
            load_lora_state(eval_model, self.global_lora_state)
            eval_model.eval()
            temp_eval_model = True
        elif self.algo == 'florg':
            if int(cur_round) <= 0:
                # Round-0 eval uses the untouched backbone baseline.
                self.model = self.model.to(self.device)
                self.model.eval()
                eval_model = self.model
            else:
                eval_model = build_florg_model(
                    deepcopy(self.model),
                    self.args,
                    seed_state=self.global_florg_seed_state,
                    basis_state=self.global_florg_basis_state,
                ).to(self.device)
                load_florg_A_state(eval_model, self.global_florg_A_state)
                eval_model.eval()
                temp_eval_model = True
                if not self._florg_eval_debug_logged:
                    layer_name, delta_norm = sample_florg_delta_norm(eval_model)
                    print(f'[debug][florg][eval] layer={layer_name} ||deltaW||={delta_norm:.6e}')
                    self._florg_eval_debug_logged = True
        elif self.algo == 'fedexlora':
            eval_model = build_lora_model(deepcopy(self.model), self.args).to(self.device)
            load_lora_A_state(eval_model, self.global_lora_A_state)
            load_lora_B_state(eval_model, self.global_lora_B_state)
            load_classifier_state(eval_model, self.global_classifier_state)
            eval_model.eval()
            temp_eval_model = True
        elif self.algo in ['fedmultisubmuon', 'fedstructmuon']:
            self.model = self.model.to(self.device)
            self.model.eval()
            eval_model = self.model
        else:
            self.model = self.model.to(self.device)
            self.model.eval()
            eval_model = self.model

        if self.algo in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd']:
            uv_state = None
            if self.algo == 'fedsubmuon_gt':
                uv_state = {'u': self.u_global, 'v': self.v_basis_global}
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                self.x_global,
                self.m_global if self.algo == 'fedsubmuon' else None,
                self.seeds,
                trainable=False,
                v_state=None,
                uv_state=uv_state,
            )
        elif self.algo == 'fedmultisubmuon':
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_multisub_state(
                {
                    'b_global': self.global_multisub_b_state,
                    'c_global': self.global_multisub_c_state,
                    'metadata': self.global_multisub_metadata,
                    'selected_keys': list(self.global_multisub_b_state.keys()),
                    'score_state': self.global_multisub_scores,
                },
                trainable=False,
            )
        elif self.algo == 'fedstructmuon':
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_struct_state(
                {
                    'x_global': self.global_struct_x_state,
                    'metadata': self.global_struct_metadata,
                    'selected_keys': list(self.global_struct_x_state.keys()),
                    'score_state': self.global_struct_scores,
                },
                trainable=False,
            )

        round_eval_loader = self._build_round_eval_loader(cur_round)
        progress_bar_eval = tqdm(range(len(round_eval_loader)))
        loss_total_eval = 0.0
        num_eval = 0

        with torch.inference_mode():
            for batch in round_eval_loader:
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
            if self.algo == 'fedmultisubmuon':
                framework.clear_multisub_state()
            elif self.algo == 'fedstructmuon':
                framework.clear_struct_state()
            else:
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
        if self.algo in ['fedit', 'federa']:
            eval_model = build_lora_model(deepcopy(self.model), self.args).to(self.device)
            load_lora_state(eval_model, self.global_lora_state)
            eval_model.eval()
            temp_eval_model = True
        elif self.algo == 'florg':
            if int(cur_round) <= 0:
                # Round-0 eval uses the untouched backbone baseline.
                self.model = self.model.to(self.device)
                self.model.eval()
                eval_model = self.model
            else:
                eval_model = build_florg_model(
                    deepcopy(self.model),
                    self.args,
                    seed_state=self.global_florg_seed_state,
                    basis_state=self.global_florg_basis_state,
                ).to(self.device)
                load_florg_A_state(eval_model, self.global_florg_A_state)
                eval_model.eval()
                temp_eval_model = True
                if not self._florg_eval_debug_logged:
                    layer_name, delta_norm = sample_florg_delta_norm(eval_model)
                    print(f'[debug][florg][eval] layer={layer_name} ||deltaW||={delta_norm:.6e}')
                    self._florg_eval_debug_logged = True
        elif self.algo == 'fedexlora':
            eval_model = build_lora_model(deepcopy(self.model), self.args).to(self.device)
            load_lora_A_state(eval_model, self.global_lora_A_state)
            load_lora_B_state(eval_model, self.global_lora_B_state)
            load_classifier_state(eval_model, self.global_classifier_state)
            eval_model.eval()
            temp_eval_model = True
        elif self.algo in ['fedmultisubmuon', 'fedstructmuon']:
            self.model = self.model.to(self.device)
            self.model.eval()
            eval_model = self.model
        else:
            self.model = self.model.to(self.device)
            self.model.eval()
            eval_model = self.model

        if self.algo in ['fedsubmuon', 'fedsubmuonv2', 'fedsubmuon_gt', 'fedsubadam', 'fedsubsgd']:
            uv_state = None
            if self.algo == 'fedsubmuon_gt':
                uv_state = {'u': self.u_global, 'v': self.v_basis_global}
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                self.x_global,
                self.m_global if self.algo == 'fedsubmuon' else None,
                self.seeds,
                trainable=False,
                v_state=None,
                uv_state=uv_state,
            )
        elif self.algo == 'fedmultisubmuon':
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_multisub_state(
                {
                    'b_global': self.global_multisub_b_state,
                    'c_global': self.global_multisub_c_state,
                    'metadata': self.global_multisub_metadata,
                    'selected_keys': list(self.global_multisub_b_state.keys()),
                    'score_state': self.global_multisub_scores,
                },
                trainable=False,
            )
        elif self.algo == 'fedstructmuon':
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_struct_state(
                {
                    'x_global': self.global_struct_x_state,
                    'metadata': self.global_struct_metadata,
                    'selected_keys': list(self.global_struct_x_state.keys()),
                    'score_state': self.global_struct_scores,
                },
                trainable=False,
            )

        round_eval_loader = self._build_round_eval_loader(cur_round)
        progress_bar_eval = tqdm(range(len(round_eval_loader)))
        if self.args.dataset == 'gsm8k':
            sanitize_greedy_generation_config(eval_model)
            pred_texts = []
            ref_texts = []
            num_eval = 0
            with torch.inference_mode():
                for batch in round_eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    label_ids = batch['labels'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    bs = input_ids.size(0)
                    for i in range(bs):
                        valid_input = input_ids[i][attention_mask[i].bool()].unsqueeze(0)
                        valid_mask = torch.ones_like(valid_input, device=self.device)
                        output_ids = eval_model.generate(
                            input_ids=valid_input,
                            attention_mask=valid_mask,
                            pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=256,
                        )
                        prompt_len = int(valid_mask[0].sum().item())
                        pred_ids = output_ids[0][prompt_len:]
                        ref_ids = label_ids[i]
                        if ref_ids.numel() > 0:
                            ref_ids = ref_ids[ref_ids >= 0]
                        pred_texts.append(self.tokenizer.decode(pred_ids, skip_special_tokens=True))
                        ref_texts.append(self.tokenizer.decode(ref_ids, skip_special_tokens=True))
                    progress_bar_eval.update(1)
                    num_eval += bs
                    progress_bar_eval.set_description(f'eval at round {cur_round}, gsm8k samples: {num_eval}')
            gsm8k_metrics = compute_gsm8k_metrics(pred_texts, ref_texts)
            print(
                f'[eval][gsm8k] round={cur_round} '
                f'acc={float(gsm8k_metrics["gsm8k_acc"]):.6f} '
                f'rougeL={float(gsm8k_metrics["gsm8k_rougeL"]):.6f} '
                f'invalid_rate={float(gsm8k_metrics["gsm8k_invalid_rate"]):.6f}'
            )
            eval_metric = float(gsm8k_metrics['gsm8k_acc'])
        elif self.args.dataset == 'math':
            sanitize_greedy_generation_config(eval_model)
            pred_texts = []
            ref_texts = []
            gold_finals = []
            subjects = []
            levels = []
            num_eval = 0
            running_correct = 0
            with torch.inference_mode():
                for batch in round_eval_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    label_ids = batch['labels'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    bs = input_ids.size(0)
                    meta_ref_solution = batch.get('meta_ref_solution', None)
                    meta_final_answer = batch.get('meta_final_answer', None)
                    meta_subject = batch.get('meta_subject', None)
                    meta_level = batch.get('meta_level', None)
                    for i in range(bs):
                        valid_input = input_ids[i][attention_mask[i].bool()].unsqueeze(0)
                        valid_mask = torch.ones_like(valid_input, device=self.device)
                        output_ids = eval_model.generate(
                            input_ids=valid_input,
                            attention_mask=valid_mask,
                            pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=512,
                        )
                        prompt_len = int(valid_mask[0].sum().item())
                        pred_ids = output_ids[0][prompt_len:]
                        pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

                        ref_text = None
                        if isinstance(meta_ref_solution, list) and i < len(meta_ref_solution):
                            ref_text = meta_ref_solution[i]
                        if ref_text is None:
                            ref_ids = label_ids[i]
                            if ref_ids.numel() > 0:
                                ref_ids = ref_ids[ref_ids >= 0]
                            ref_text = self.tokenizer.decode(ref_ids, skip_special_tokens=True)

                        gold_final = None
                        if isinstance(meta_final_answer, list) and i < len(meta_final_answer):
                            gold_final = meta_final_answer[i]
                        subject = 'unknown'
                        if isinstance(meta_subject, list) and i < len(meta_subject) and meta_subject[i] is not None:
                            subject = str(meta_subject[i])
                        level = 'unknown'
                        if isinstance(meta_level, list) and i < len(meta_level) and meta_level[i] is not None:
                            level = str(meta_level[i])

                        pred_texts.append(pred_text)
                        ref_texts.append(ref_text)
                        gold_finals.append(gold_final)
                        subjects.append(subject)
                        levels.append(level)

                        pred_final, pred_invalid = extract_math_pred_final_answer(pred_text)
                        gold_final_norm = extract_math_gold_final_answer(gold_final)
                        if (not pred_invalid) and (pred_final is not None) and (gold_final_norm is not None) and (pred_final == gold_final_norm):
                            running_correct += 1
                    progress_bar_eval.update(1)
                    num_eval += bs
                    denom = float(max(num_eval, 1))
                    progress_bar_eval.set_description(f'eval at round {cur_round}, math_acc: {running_correct / denom:.6f}')
            math_metrics = compute_math_metrics(
                pred_texts=pred_texts,
                ref_texts=ref_texts,
                gold_finals=gold_finals,
                subjects=subjects,
                levels=levels,
            )
            print(
                f'[eval][math] round={cur_round} '
                f'acc={float(math_metrics["math_acc"]):.6f} '
                f'rougeL={float(math_metrics["math_rougeL"]):.6f} '
                f'invalid_rate={float(math_metrics["math_invalid_rate"]):.6f}'
            )
            print(
                f'[eval][math] round={cur_round} '
                f'acc_by_subject={math_metrics.get("math_acc_by_subject", {})} '
                f'acc_by_level={math_metrics.get("math_acc_by_level", {})}'
            )
            eval_metric = float(math_metrics['math_acc'])
        else:
            acc_total_eval = 0.0
            num_eval = 0
            with torch.inference_mode():
                for batch in round_eval_loader:
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
            eval_metric = acc_total_eval / num_eval
        print()
        print()

        if framework is not None:
            if self.algo == 'fedmultisubmuon':
                framework.clear_multisub_state()
            elif self.algo == 'fedstructmuon':
                framework.clear_struct_state()
            else:
                framework.clear_submuon_state()
        if temp_eval_model:
            eval_model = eval_model.cpu()
            del eval_model
        self.model = self.model.cpu()
        return eval_metric
