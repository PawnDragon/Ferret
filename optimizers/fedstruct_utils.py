from typing import Dict, List, Tuple

import torch

from optimizers.fedmultisub_utils import partition_subspaces
from optimizers.submuon_utils import make_uv, select_target_linear_layers


def initialize_struct_subspaces(
    model,
    rank_r: int,
    svd_rank: int,
    num_subspaces: int,
    base_seed: int,
    target_modules=None,
) -> Tuple[Dict[str, dict], Dict[str, torch.Tensor], Dict[str, float]]:
    """
    Initialize FedStructMuon subspaces:
    fixed A_k from one-time structured decomposition + fixed V_k from FedSubMuon helper.
    Trainable core X_k is initialized to zeros so initial delta is exactly zero.
    """
    module_map = dict(model.named_modules())
    layer_names = select_target_linear_layers(model, rank_r, raw_target_modules=target_modules)
    if len(layer_names) == 0:
        raise RuntimeError(
            f'[fedstructmuon] no target linear layer is selected; '
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
            n_sub = int(col_indices.numel())
            if n_sub == 0:
                continue

            w_sub = weight[:, col_indices]
            u, _, _ = torch.linalg.svd(w_sub, full_matrices=False)
            rank_local = int(min(rank_r, int(u.shape[1]), n_sub))
            if rank_local <= 0:
                continue

            key = f'{layer_name}::sub{sub_idx}'
            seed = int(base_seed + layer_idx * 1000003 + sub_idx * 104729)
            _, v_basis = make_uv(
                seed=seed,
                out_dim=n_sub,
                in_dim=n_sub,
                r=rank_local,
                device='cpu',
                dtype=torch.float32,
            )
            a_tensor = u[:, :rank_local].contiguous().cpu()
            v_tensor = v_basis.cpu().contiguous()
            x_tensor = torch.zeros((rank_local, rank_local), dtype=torch.float32)

            metadata[key] = {
                'layer_name': str(layer_name),
                'indices': col_indices.to(dtype=torch.long).cpu().contiguous(),
                'A': a_tensor,
                'V': v_tensor,
                'rank': int(rank_local),
                'seed': int(seed),
                'flat_id': int(flat_id),
            }
            x_state[key] = x_tensor
            score_state[key] = 0.0
            flat_id += 1

    if len(metadata) == 0:
        raise RuntimeError('[fedstructmuon] failed to initialize any subspace')
    return metadata, x_state, score_state


def shape_signature(meta_entry: dict, x_tensor: torch.Tensor) -> str:
    a_shape = tuple(meta_entry['A'].shape) if isinstance(meta_entry.get('A', None), torch.Tensor) else None
    v_shape = tuple(meta_entry['V'].shape) if isinstance(meta_entry.get('V', None), torch.Tensor) else None
    x_shape = tuple(x_tensor.shape) if isinstance(x_tensor, torch.Tensor) else None
    cols = int(meta_entry['indices'].numel()) if isinstance(meta_entry.get('indices', None), torch.Tensor) else -1
    return f'A={a_shape}, X={x_shape}, V={v_shape}, cols={cols}'
