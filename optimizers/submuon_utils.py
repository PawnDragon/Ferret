import torch
import torch.nn as nn


DEFAULT_TARGET_MODULES = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']


def zeropower_via_newtonschulz5(G, steps):
    """
    Compute matrix zeroth-power direction with a 5th-order Newton-Schulz iteration.
    The routine follows Muon-style usage and is stable for small square cores.
    """
    X = G.to(torch.bfloat16)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True

    eps = 1e-7
    X = X / (torch.linalg.norm(X) + eps)

    a = 3.4445
    b = -4.7750
    c = 2.0315

    for _ in range(steps):
        A = X @ X.t()
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.t()
    return X


def make_uv(seed, out_dim, in_dim, r, device, dtype):
    """
    Create deterministic orthonormal column bases U(out_dim, r), V(in_dim, r)
    from a seed using CPU RNG + QR, then move to target device/dtype.
    """
    rank = min(r, out_dim, in_dim)
    g_u = torch.Generator(device='cpu')
    g_v = torch.Generator(device='cpu')
    g_u.manual_seed(int(seed))
    g_v.manual_seed(int(seed) + 1)

    U = torch.randn(out_dim, rank, generator=g_u, dtype=torch.float32)
    V = torch.randn(in_dim, rank, generator=g_v, dtype=torch.float32)

    U = torch.linalg.qr(U, mode='reduced').Q
    V = torch.linalg.qr(V, mode='reduced').Q

    U = U.to(device=device, dtype=dtype)
    V = V.to(device=device, dtype=dtype)
    return U, V


def _parse_target_modules(raw_target_modules):
    if raw_target_modules is None:
        return 'all-linear'
    if isinstance(raw_target_modules, (list, tuple)):
        parsed = [str(item).strip() for item in raw_target_modules if str(item).strip() != '']
        return parsed if len(parsed) > 0 else 'all-linear'

    target_text = str(raw_target_modules).strip()
    if target_text == '':
        return 'all-linear'
    if target_text.lower() in ['all-linear', 'all_linear']:
        return 'all-linear'
    if ',' in target_text:
        parsed = [part.strip() for part in target_text.split(',') if part.strip() != '']
        return parsed if len(parsed) > 0 else DEFAULT_TARGET_MODULES
    return [target_text]


def _is_target_layer_name(layer_name, target_modules):
    if target_modules == 'all-linear':
        return True
    leaf_name = layer_name.split('.')[-1]
    for token in target_modules:
        if layer_name == token or layer_name.endswith(f'.{token}') or leaf_name == token:
            return True
    return False


def select_target_linear_layers(model, rank, raw_target_modules=None):
    all_targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if min(module.in_features, module.out_features) >= rank:
                all_targets.append(name)

    target_modules = _parse_target_modules(raw_target_modules)
    if target_modules == 'all-linear':
        return all_targets
    return [name for name in all_targets if _is_target_layer_name(name, target_modules)]


def init_submuon_state(model, rank, base_seed, raw_target_modules=None):
    layer_names = select_target_linear_layers(model, rank, raw_target_modules=raw_target_modules)
    if len(layer_names) == 0:
        raise RuntimeError(
            f'[fedsub] no target linear layer is selected; '
            f'rank_r={rank}, lora_target_modules={raw_target_modules}'
        )
    x_global = {}
    m_global = {}
    seeds = {}
    for idx, name in enumerate(layer_names):
        x_global[name] = torch.zeros(rank, rank, dtype=torch.float32)
        m_global[name] = torch.zeros(rank, rank, dtype=torch.float32)
        seeds[name] = int(base_seed + idx * 104729)
    return x_global, m_global, seeds


def transport_state(x_global, m_global, old_seeds, new_seeds, layer_dims, rank, v_global=None):
    for name, (out_dim, in_dim) in layer_dims.items():
        U_old, V_old = make_uv(old_seeds[name], out_dim, in_dim, rank, device='cpu', dtype=torch.float32)
        U_new, V_new = make_uv(new_seeds[name], out_dim, in_dim, rank, device='cpu', dtype=torch.float32)
        R_u = U_new.t() @ U_old
        R_v = V_new.t() @ V_old
        x_global[name] = R_u @ x_global[name] @ R_v.t()
        if m_global is not None:
            m_global[name] = R_u @ m_global[name] @ R_v.t()
        if v_global is not None:
            v_global[name] = R_u @ v_global[name] @ R_v.t()


def fold_submuon_core_into_backbone(model, x_state, seeds, rank, layer_dims=None):
    if not isinstance(x_state, dict) or len(x_state) == 0:
        return {'num_layers': 0, 'delta_norm': 0.0}
    if not isinstance(seeds, dict):
        raise RuntimeError('[fedsubmuonv2] seeds must be a dict when folding UXV^T into backbone')

    name_to_module = dict(model.named_modules())
    folded_layers = 0
    total_delta_norm = 0.0

    with torch.no_grad():
        for layer_name, x_tensor in x_state.items():
            if layer_name not in seeds:
                raise RuntimeError(f'[fedsubmuonv2] missing seed for layer={layer_name}')
            if layer_name not in name_to_module:
                raise RuntimeError(f'[fedsubmuonv2] layer not found in model: {layer_name}')
            if not isinstance(x_tensor, torch.Tensor):
                raise RuntimeError(f'[fedsubmuonv2] expected tensor X for layer={layer_name}')

            module = name_to_module[layer_name]
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f'[fedsubmuonv2] layer={layer_name} is not nn.Linear')

            if isinstance(layer_dims, dict) and layer_name in layer_dims:
                out_dim, in_dim = layer_dims[layer_name]
            else:
                out_dim, in_dim = int(module.out_features), int(module.in_features)

            x_local = x_tensor.to(device=module.weight.device, dtype=torch.float32)
            U, V = make_uv(
                seed=seeds[layer_name],
                out_dim=int(out_dim),
                in_dim=int(in_dim),
                r=int(rank),
                device=module.weight.device,
                dtype=torch.float32,
            )

            expected_x_shape = (int(U.shape[1]), int(V.shape[1]))
            if tuple(x_local.shape) != expected_x_shape:
                raise RuntimeError(
                    f'[fedsubmuonv2] X shape mismatch @ {layer_name}: '
                    f'got={tuple(x_local.shape)}, expected={expected_x_shape}'
                )

            delta = U @ x_local @ V.t()
            if tuple(delta.shape) != tuple(module.weight.shape):
                raise RuntimeError(
                    f'[fedsubmuonv2] UXV^T shape mismatch @ {layer_name}: '
                    f'got={tuple(delta.shape)}, expected={tuple(module.weight.shape)}'
                )

            module.weight.data.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
            total_delta_norm += float(torch.linalg.norm(delta.float()).item())
            folded_layers += 1

    return {'num_layers': int(folded_layers), 'delta_norm': float(total_delta_norm)}


def relative_transport_error(X_old, old_seed, new_seed, out_dim, in_dim, rank):
    U_old, V_old = make_uv(old_seed, out_dim, in_dim, rank, device='cpu', dtype=torch.float32)
    U_new, V_new = make_uv(new_seed, out_dim, in_dim, rank, device='cpu', dtype=torch.float32)
    R_u = U_new.t() @ U_old
    R_v = V_new.t() @ V_old
    X_new = R_u @ X_old @ R_v.t()

    W_old = U_old @ X_old @ V_old.t()
    W_new = U_new @ X_new @ V_new.t()

    denom = torch.linalg.norm(W_old) + 1e-12
    return (torch.linalg.norm(W_old - W_new) / denom).item()


def estimate_comm_bytes(num_layers, rank, num_clients, include_seed=True, num_state_tensors=2):
    per_layer_tensor = rank * rank * 4
    per_client_up = num_layers * per_layer_tensor * num_state_tensors
    per_client_down = num_layers * per_layer_tensor * num_state_tensors
    if include_seed:
        per_client_down += num_layers * 8
    return per_client_up * num_clients, per_client_down * num_clients
