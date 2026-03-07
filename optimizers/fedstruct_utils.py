from typing import Dict, List, Tuple

import torch

from optimizers.fedmultisub_utils import partition_subspaces
from optimizers.submuon_utils import make_uv, select_target_linear_layers


def initialize_struct_subspaces(
    model,
    rank_r: int = None,
    rank_left: int = None,
    rank_right: int = None,
    svd_rank: int = 500,
    num_subspaces: int = 4,
    base_seed: int = 0,
    target_modules=None,
) -> Tuple[Dict[str, dict], Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Initialize FedStructMuon subspaces:
    fixed A_k from one-time structured decomposition + fixed V_k from FedSubMuon helper.
    Trainable core X_k is initialized to zeros so initial delta is exactly zero.
    """
    if rank_left is None:
        rank_left = rank_r
    if rank_right is None:
        rank_right = rank_r
    if rank_left is None or rank_right is None:
        raise RuntimeError('[fedstructmuon] rank_left/rank_right cannot both be None')
    rank_left = int(rank_left)
    rank_right = int(rank_right)
    if rank_left <= 0 or rank_right <= 0:
        raise RuntimeError(
            f'[fedstructmuon] invalid ranks: rank_left={rank_left}, rank_right={rank_right}'
        )

    module_map = dict(model.named_modules())
    # Keep layer filtering behavior compatible with legacy square-rank mode.
    layer_names = select_target_linear_layers(
        model,
        min(rank_left, rank_right),
        raw_target_modules=target_modules,
    )
    if len(layer_names) == 0:
        raise RuntimeError(
            f'[fedstructmuon] no target linear layer is selected; '
            f'rank_left={rank_left}, rank_right={rank_right}, '
            f'svd_rank={svd_rank}, lora_target_modules={target_modules}'
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
        out_dim = int(weight.shape[0])
        in_dim = int(weight.shape[1])
        subspace_columns = partition_subspaces(
            weight,
            num_subspaces=num_subspaces,
            spectral_rank=int(svd_rank),
        )
        layer_debug_rows = []

        for sub_idx, col_indices in enumerate(subspace_columns):
            n_sub = int(col_indices.numel())
            if n_sub == 0:
                continue

            w_sub = weight[:, col_indices]
            u, _, _ = torch.linalg.svd(w_sub, full_matrices=False)
            rank_left_local = int(min(rank_left, int(u.shape[1])))
            rank_right_local = int(min(rank_right, n_sub))
            if rank_left_local <= 0 or rank_right_local <= 0:
                continue

            key = f'{layer_name}::sub{sub_idx}'
            seed = int(base_seed + layer_idx * 1000003 + sub_idx * 104729)
            _, v_basis = make_uv(
                seed=seed,
                out_dim=n_sub,
                in_dim=n_sub,
                r=rank_right_local,
                device='cpu',
                dtype=torch.float32,
            )
            a_tensor = u[:, :rank_left_local].contiguous().cpu()
            v_tensor = v_basis.cpu().contiguous()
            x_tensor = torch.zeros((rank_left_local, rank_right_local), dtype=torch.float32)

            metadata[key] = {
                'layer_name': str(layer_name),
                'indices': col_indices.to(dtype=torch.long).cpu().contiguous(),
                'A': a_tensor,
                'V': v_tensor,
                'rank': int(rank_left_local),  # backward-compatible alias
                'rank_left': int(rank_left_local),
                'rank_right': int(rank_right_local),
                'seed': int(seed),
                'flat_id': int(flat_id),
            }
            x_state[key] = x_tensor
            score_state[key] = 0.0
            flat_id += 1
            layer_debug_rows.append(
                (
                    sub_idx,
                    n_sub,
                    rank_left_local,
                    rank_right_local,
                )
            )

        if len(layer_debug_rows) > 0:
            detail_text = ", ".join(
                [
                    (
                        f"sub{sub_idx}: block=({out_dim},{n_sub}), "
                        f"A=({out_dim},{r_left}), X=({r_left},{r_right}), V=({n_sub},{r_right})"
                    )
                    for sub_idx, n_sub, r_left, r_right in layer_debug_rows
                ]
            )
            print(
                f"[fedstructmuon][init] layer={layer_name} W=({out_dim},{in_dim}) "
                f"splits={len(layer_debug_rows)} -> {detail_text}"
            )

    if len(metadata) == 0:
        raise RuntimeError('[fedstructmuon] failed to initialize any subspace')
    return metadata, x_state, score_state


def shape_signature(meta_entry: dict, x_tensor: torch.Tensor) -> str:
    a_shape = tuple(meta_entry['A'].shape) if isinstance(meta_entry.get('A', None), torch.Tensor) else None
    v_shape = tuple(meta_entry['V'].shape) if isinstance(meta_entry.get('V', None), torch.Tensor) else None
    x_shape = tuple(x_tensor.shape) if isinstance(x_tensor, torch.Tensor) else None
    cols = int(meta_entry['indices'].numel()) if isinstance(meta_entry.get('indices', None), torch.Tensor) else -1
    return f'A={a_shape}, X={x_shape}, V={v_shape}, cols={cols}'
