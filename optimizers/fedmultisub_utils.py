import math
from typing import Dict, List, Tuple

import torch

from optimizers.submuon_utils import select_target_linear_layers


def _split_indices_evenly(indices: torch.Tensor, num_parts: int) -> List[torch.Tensor]:
    if num_parts <= 0:
        return [indices]
    n = int(indices.numel())
    if n == 0:
        return []
    num_parts = min(int(num_parts), n)
    parts = []
    base = n // num_parts
    rem = n % num_parts
    start = 0
    for i in range(num_parts):
        width = base + (1 if i < rem else 0)
        end = start + width
        if end > start:
            parts.append(indices[start:end].clone())
        start = end
    return parts


def partition_subspaces(weight: torch.Tensor, num_subspaces: int, spectral_rank: int = 500) -> List[torch.Tensor]:
    # AdaMSS-inspired spectral partition: sort columns by top singular feature, then split evenly.
    in_dim = int(weight.shape[1])
    if in_dim == 0:
        return []
    if num_subspaces <= 1:
        return [torch.arange(in_dim, dtype=torch.long)]

    rank = int(max(1, min(spectral_rank, min(weight.shape[0], weight.shape[1]))))
    _, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    features = vh[:rank, :].t().contiguous()
    order = torch.argsort(features[:, 0], descending=True)
    return _split_indices_evenly(order.to(dtype=torch.long), int(num_subspaces))


def initialize_subspaces(
    model,
    rank_r: int,
    svd_rank: int,
    num_subspaces: int,
    base_seed: int,
    target_modules=None,
) -> Tuple[Dict[str, dict], Dict[str, torch.Tensor], Dict[str, float]]:
    module_map = dict(model.named_modules())
    layer_names = select_target_linear_layers(model, rank_r, raw_target_modules=target_modules)
    if len(layer_names) == 0:
        raise RuntimeError(
            f'[fedmultisubmuon] no target linear layer is selected; '
            f'rank_r={rank_r}, svd_rank={svd_rank}, lora_target_modules={target_modules}'
        )

    metadata = {}
    x_state = {}
    score_state = {}
    flat_id = 0

    for layer_idx, layer_name in enumerate(layer_names):
        module = module_map.get(layer_name, None)
        if module is None or (not hasattr(module, 'weight')):
            continue
        weight = module.weight.detach().cpu().float()
        subspace_columns = partition_subspaces(
            weight,
            num_subspaces=num_subspaces,
            spectral_rank=int(svd_rank),
        )
        for sub_idx, col_indices in enumerate(subspace_columns):
            if int(col_indices.numel()) == 0:
                continue
            w_sub = weight[:, col_indices]
            u, _, _ = torch.linalg.svd(w_sub, full_matrices=False)
            rank_local = int(min(rank_r, u.shape[1], int(col_indices.numel())))
            if rank_local <= 0:
                continue
            key = f'{layer_name}::sub{sub_idx}'
            seed = int(base_seed + layer_idx * 1000003 + sub_idx * 104729)
            metadata[key] = {
                'layer_name': layer_name,
                'indices': col_indices.to(dtype=torch.long).cpu().contiguous(),
                'A': u[:, :rank_local].contiguous().cpu(),
                'seed': seed,
                'rank': rank_local,
                'flat_id': int(flat_id),
            }
            x_state[key] = torch.zeros(rank_local, rank_local, dtype=torch.float32)
            score_state[key] = 0.0
            flat_id += 1

    if len(metadata) == 0:
        raise RuntimeError('[fedmultisubmuon] failed to initialize any subspace')
    return metadata, x_state, score_state


def select_topk_subspaces(score_state: Dict[str, float], topk: int) -> List[str]:
    if len(score_state) == 0:
        return []
    k = int(topk)
    items = sorted(score_state.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    if k <= 0 or k >= len(items):
        return [key for key, _ in items]
    return [key for key, _ in items[:k]]


def update_adamss_score(
    ipt_abs_scalar: float,
    exp_avg_prev: float,
    exp_unc_prev: float,
    beta1: float,
    beta2: float,
) -> Tuple[float, float, float]:
    # AdaMSS scoring adapted from asa.py: score = exp_avg_ipt * exp_avg_unc.
    exp_avg = float(beta1) * float(exp_avg_prev) + (1.0 - float(beta1)) * float(ipt_abs_scalar)
    exp_unc = float(beta2) * float(exp_unc_prev) + (1.0 - float(beta2)) * abs(float(ipt_abs_scalar) - exp_avg)
    score = exp_avg * exp_unc
    return exp_avg, exp_unc, score


def orthogonality_error(v_mat: torch.Tensor) -> float:
    if v_mat.ndim != 2:
        return float('nan')
    eye = torch.eye(v_mat.shape[1], device=v_mat.device, dtype=v_mat.dtype)
    gram = v_mat.t().matmul(v_mat)
    return float(torch.linalg.norm(gram - eye).item())


def shape_signature(meta_entry: dict, x_tensor: torch.Tensor) -> str:
    a_shape = tuple(meta_entry['A'].shape) if isinstance(meta_entry.get('A', None), torch.Tensor) else None
    x_shape = tuple(x_tensor.shape) if isinstance(x_tensor, torch.Tensor) else None
    cols = int(meta_entry['indices'].numel()) if isinstance(meta_entry.get('indices', None), torch.Tensor) else -1
    return f'A={a_shape}, X={x_shape}, cols={cols}, rank={meta_entry.get("rank", None)}'
