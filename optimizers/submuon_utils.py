import torch
import torch.nn as nn


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


def select_target_linear_layers(model, rank):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if min(module.in_features, module.out_features) >= rank:
                targets.append(name)
    return targets


def init_submuon_state(model, rank, base_seed):
    layer_names = select_target_linear_layers(model, rank)
    x_global = {}
    m_global = {}
    seeds = {}
    for idx, name in enumerate(layer_names):
        x_global[name] = torch.zeros(rank, rank, dtype=torch.float32)
        m_global[name] = torch.zeros(rank, rank, dtype=torch.float32)
        seeds[name] = int(base_seed + idx * 104729)
    return x_global, m_global, seeds


def transport_state(x_global, m_global, old_seeds, new_seeds, layer_dims, rank):
    for name, (out_dim, in_dim) in layer_dims.items():
        U_old, V_old = make_uv(old_seeds[name], out_dim, in_dim, rank, device='cpu', dtype=torch.float32)
        U_new, V_new = make_uv(new_seeds[name], out_dim, in_dim, rank, device='cpu', dtype=torch.float32)
        R_u = U_new.t() @ U_old
        R_v = V_new.t() @ V_old
        x_global[name] = R_u @ x_global[name] @ R_v.t()
        m_global[name] = R_u @ m_global[name] @ R_v.t()


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


def estimate_comm_bytes(num_layers, rank, num_clients, include_seed=True):
    per_layer_tensor = rank * rank * 4
    per_client_up = num_layers * per_layer_tensor * 2
    per_client_down = num_layers * per_layer_tensor * 2
    if include_seed:
        per_client_down += num_layers * 8
    return per_client_up * num_clients, per_client_down * num_clients
