from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from optimizers.adamss_allocator import SubspacesAllocator
from optimizers.submuon_utils import select_target_linear_layers


def _orthogonal_row_init(rows: int, cols: int, seed: int) -> torch.Tensor:
    """
    Return a (rows, cols) matrix with orthonormal rows (rows <= cols).
    This mirrors AdaMSS orth-style initialization where one factor is orthogonal.
    """
    if rows <= 0 or cols <= 0:
        return torch.zeros((rows, cols), dtype=torch.float32)
    if rows > cols:
        raise RuntimeError(f'[fedmultisubmuon] orth init requires rows<=cols, got rows={rows}, cols={cols}')
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    # QR on (cols, rows) gives column-orthonormal Q; transpose -> row-orthonormal.
    rand = torch.randn((cols, rows), generator=gen, dtype=torch.float32)
    q, _ = torch.linalg.qr(rand, mode='reduced')
    return q.t().contiguous()


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
    # AdaMSS-style spectral features from top singular vectors, then even split.
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
) -> Tuple[Dict[str, dict], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, float]]:
    module_map = dict(model.named_modules())
    layer_names = select_target_linear_layers(model, rank_r, raw_target_modules=target_modules)
    if len(layer_names) == 0:
        raise RuntimeError(
            f'[fedmultisubmuon] no target linear layer is selected; '
            f'rank_r={rank_r}, svd_rank={svd_rank}, lora_target_modules={target_modules}'
        )

    metadata = {}
    b_state = {}
    c_state = {}
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
            n_sub = int(col_indices.numel())
            if n_sub == 0:
                continue
            w_sub = weight[:, col_indices]
            u, _, _ = torch.linalg.svd(w_sub, full_matrices=False)
            rank_big = int(min(rank_r, int(u.shape[1])))
            if rank_big <= 0:
                continue
            rank_small = int(min(rank_big, n_sub))
            if rank_small <= 0:
                continue

            key = f'{layer_name}::sub{sub_idx}'
            seed = int(base_seed + layer_idx * 1000003 + sub_idx * 104729)

            a_tensor = u[:, :rank_big].contiguous().cpu()
            b_tensor = torch.zeros((rank_big, rank_small), dtype=torch.float32)
            c_tensor = _orthogonal_row_init(rank_small, n_sub, seed=seed).cpu()

            metadata[key] = {
                'layer_name': layer_name,
                'indices': col_indices.to(dtype=torch.long).cpu().contiguous(),
                'A': a_tensor,
                'seed': seed,
                'rank_big': rank_big,
                'rank_small': rank_small,
                'flat_id': int(flat_id),
            }
            b_state[key] = b_tensor
            c_state[key] = c_tensor
            score_state[key] = 0.0
            flat_id += 1

    if len(metadata) == 0:
        raise RuntimeError('[fedmultisubmuon] failed to initialize any subspace')
    return metadata, b_state, c_state, score_state


def select_topk_subspaces(score_state: Dict[str, float], topk: int) -> List[str]:
    if len(score_state) == 0:
        return []
    k = int(topk)
    items = sorted(score_state.items(), key=lambda kv: (-float(kv[1]), kv[0]))
    if k <= 0 or k >= len(items):
        return [key for key, _ in items]
    return [key for key, _ in items[:k]]


def build_adamss_allocator(mask_interval: int, beta1: float, beta2: float):
    b1 = float(max(min(beta1, 1.0 - 1e-6), 1e-6))
    b2 = float(max(min(beta2, 1.0 - 1e-6), 1e-6))
    return SubspacesAllocator(
        tt=1.0,
        target_KK=1,
        init_warmup=0,
        final_warmup=10**9,
        mask_interval=max(int(mask_interval), 1),
        beta1=b1,
        beta2=b2,
        total_step=10**9 + 1,
    )


class _AdaMSSScoreProxy(nn.Module):
    def __init__(self, prefix: str, b_value: torch.Tensor, b_grad: torch.Tensor, c_value: torch.Tensor, c_grad: torch.Tensor):
        super().__init__()
        self._prefix = str(prefix)
        self._param_a = nn.Parameter(b_value.detach().clone(), requires_grad=False)
        self._param_b = nn.Parameter(c_value.detach().clone(), requires_grad=False)
        self._param_a.grad = b_grad.detach().clone()
        self._param_b.grad = c_grad.detach().clone()

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        yield f'{self._prefix}.adamss_A', self._param_a
        yield f'{self._prefix}.adamss_B', self._param_b


def compute_adamss_subspace_score(allocator, sub_key: str, b_param: torch.Tensor, c_param: torch.Tensor):
    if allocator is None or b_param.grad is None or c_param.grad is None:
        return None
    proxy = _AdaMSSScoreProxy(
        prefix=sub_key,
        b_value=b_param.detach().float().cpu(),
        b_grad=b_param.grad.detach().float().cpu(),
        c_value=c_param.detach().float().cpu(),
        c_grad=c_param.grad.detach().float().cpu(),
    )
    allocator.update_ipt(proxy)
    score_a = allocator.calculate_score(f'{sub_key}.adamss_A', metric='ipt')
    score_b = allocator.calculate_score(f'{sub_key}.adamss_B', metric='ipt')
    return float(torch.mean(score_a).item() + torch.mean(score_b).item())


def shape_signature(meta_entry: dict, b_tensor: torch.Tensor, c_tensor: torch.Tensor) -> str:
    a_shape = tuple(meta_entry['A'].shape) if isinstance(meta_entry.get('A', None), torch.Tensor) else None
    b_shape = tuple(b_tensor.shape) if isinstance(b_tensor, torch.Tensor) else None
    c_shape = tuple(c_tensor.shape) if isinstance(c_tensor, torch.Tensor) else None
    cols = int(meta_entry['indices'].numel()) if isinstance(meta_entry.get('indices', None), torch.Tensor) else -1
    return f'A={a_shape}, B={b_shape}, C={c_shape}, cols={cols}'
