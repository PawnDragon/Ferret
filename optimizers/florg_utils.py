import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from optimizers.submuon_utils import make_uv, select_target_linear_layers


FLORG_A_TOKEN = ".florg_A"


def _resolve_module(model, module_name):
    module = model
    for part in module_name.split("."):
        module = getattr(module, part)
    return module


def _clone_seed_state(seed_state):
    if not isinstance(seed_state, dict):
        return {"base_seed": None, "layer_seeds": {}}
    layer_seeds = seed_state.get("layer_seeds", {})
    if not isinstance(layer_seeds, dict):
        layer_seeds = {}
    return {
        "base_seed": seed_state.get("base_seed", None),
        "layer_seeds": {str(k): int(v) for k, v in layer_seeds.items()},
    }


def _parse_target_modules(raw_target_modules):
    if raw_target_modules is None:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if isinstance(raw_target_modules, (list, tuple)):
        parsed = [str(item).strip() for item in raw_target_modules if str(item).strip() != ""]
        return parsed if len(parsed) > 0 else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    target_text = str(raw_target_modules).strip()
    if target_text == "":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if target_text.lower() in ["all-linear", "all_linear"]:
        return "all-linear"
    if "," in target_text:
        parsed = [part.strip() for part in target_text.split(",") if part.strip() != ""]
        return parsed if len(parsed) > 0 else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return [target_text]


def select_florg_target_linear_layers(model, args, rank_r):
    all_linear_layers = select_target_linear_layers(model, rank_r)
    target_modules = _parse_target_modules(getattr(args, "lora_target_modules", None))
    if target_modules == "all-linear":
        return all_linear_layers

    selected = []
    for layer_name in all_linear_layers:
        leaf_name = layer_name.split(".")[-1]
        for token in target_modules:
            if layer_name == token or layer_name.endswith(f".{token}") or leaf_name == token:
                selected.append(layer_name)
                break
    return selected


def _clone_basis_state(basis_state):
    if not isinstance(basis_state, dict):
        return {}
    out = {}
    for layer_name, layer_state in basis_state.items():
        if not isinstance(layer_state, dict):
            continue
        l_tensor = layer_state.get("L", None)
        r_tensor = layer_state.get("R", None)
        if not isinstance(l_tensor, torch.Tensor) or not isinstance(r_tensor, torch.Tensor):
            continue
        out[str(layer_name)] = {
            "L": l_tensor.detach().cpu().clone(),
            "R": r_tensor.detach().cpu().clone(),
        }
    return out


def _make_layer_seed_state(layer_names, base_seed):
    layer_seeds = {}
    for idx, layer_name in enumerate(layer_names):
        layer_seeds[layer_name] = int(base_seed + idx * 104729)
    return {"base_seed": int(base_seed), "layer_seeds": layer_seeds}


def _init_florg_a(seed, rank_r, k):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 2027)
    std = 1.0 / math.sqrt(max(k, 1))
    return torch.randn(rank_r, k, generator=generator, dtype=torch.float32) * std


def build_florg_model(backbone_model, args, seed_state=None, basis_state=None):
    rank_r = int(getattr(args, "florg_rank_r", getattr(args, "rank_r", 8)))
    if rank_r <= 0:
        raise ValueError(f"florg rank must be positive, got {rank_r}")
    target_layer_names = select_florg_target_linear_layers(backbone_model, args, rank_r)
    if len(target_layer_names) == 0:
        raise RuntimeError(
            "[florg] no target linear layers matched. "
            "Please check --lora_target_modules/--florg_rank_r."
        )

    if seed_state is None:
        base_seed = int(getattr(args, "florg_seed_base", int(getattr(args, "seed", 42)) + 911))
        seed_state = _make_layer_seed_state(target_layer_names, base_seed)
    else:
        seed_state = _clone_seed_state(seed_state)
    if not isinstance(basis_state, dict):
        basis_state = {}

    layer_seeds = seed_state.get("layer_seeds", {})
    resolved_basis_state = {}
    for layer_name in target_layer_names:
        if layer_name not in layer_seeds:
            raise RuntimeError(f"[florg] missing layer seed for {layer_name}")

        module = _resolve_module(backbone_model, layer_name)
        if not isinstance(module, nn.Linear):
            continue

        out_dim = int(module.out_features)
        in_dim = int(module.in_features)
        k = int(min(out_dim, in_dim))
        if rank_r > k:
            raise RuntimeError(
                f"[florg] rank mismatch for {layer_name}: rank_r={rank_r}, k={k}, "
                f"in={in_dim}, out={out_dim}"
            )

        basis_entry = basis_state.get(layer_name, {})
        basis_l = basis_entry.get("L", None) if isinstance(basis_entry, dict) else None
        basis_r = basis_entry.get("R", None) if isinstance(basis_entry, dict) else None

        if isinstance(basis_l, torch.Tensor) and isinstance(basis_r, torch.Tensor):
            L = basis_l.to(device=module.weight.device, dtype=torch.float32).contiguous()
            R = basis_r.to(device=module.weight.device, dtype=torch.float32).contiguous()
        else:
            U, V = make_uv(
                seed=int(layer_seeds[layer_name]),
                out_dim=out_dim,
                in_dim=in_dim,
                r=k,
                device=module.weight.device,
                dtype=torch.float32,
            )
            L = U.contiguous()
            R = V.t().contiguous()

        expected_l_shape = (out_dim, k)
        expected_r_shape = (k, in_dim)
        if tuple(L.shape) != expected_l_shape or tuple(R.shape) != expected_r_shape:
            raise RuntimeError(
                f"[florg] basis shape mismatch for {layer_name}: "
                f"L={tuple(L.shape)} expected={expected_l_shape}, "
                f"R={tuple(R.shape)} expected={expected_r_shape}"
            )
        A_init = _init_florg_a(int(layer_seeds[layer_name]), rank_r, k)

        if hasattr(module, "florg_A"):
            delattr(module, "florg_A")
        if hasattr(module, "florg_L"):
            delattr(module, "florg_L")
        if hasattr(module, "florg_R"):
            delattr(module, "florg_R")

        module.register_parameter("florg_A", nn.Parameter(A_init.to(device=module.weight.device), requires_grad=True))
        module.register_buffer("florg_L", L, persistent=True)
        module.register_buffer("florg_R", R, persistent=True)
        resolved_basis_state[layer_name] = {
            "L": L.detach().cpu().clone(),
            "R": R.detach().cpu().clone(),
        }

        if not hasattr(module, "_florg_orig_forward"):
            module._florg_orig_forward = module.forward

        def _patched_forward(x, _module=module):
            input_shape = x.shape
            x2 = x.reshape(-1, input_shape[-1])
            y0 = F.linear(x2, _module.weight, _module.bias)

            A = _module.florg_A.float()
            Q = torch.matmul(A.t(), A)
            # Equivalent to F.linear(x2, (L @ Q @ R), None) but avoids materializing deltaW.
            z = torch.matmul(x2.float(), _module.florg_R.float().t())
            z = torch.matmul(z, Q)
            y1 = torch.matmul(z, _module.florg_L.float().t())
            y = y0.float() + y1
            return y.to(dtype=y0.dtype).reshape(*input_shape[:-1], _module.out_features)

        module.forward = _patched_forward

    backbone_model._florg_seed_state = _clone_seed_state(seed_state)
    backbone_model._florg_basis_state = _clone_basis_state(resolved_basis_state)
    backbone_model._florg_rank_r = int(rank_r)
    return backbone_model


def extract_florg_A_state(model):
    out = {}
    for name, param in model.named_parameters():
        if FLORG_A_TOKEN not in name:
            continue
        out[name] = param.detach().cpu().clone()
    return out


def load_florg_A_state(model, florg_A_state):
    if florg_A_state is None:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if FLORG_A_TOKEN not in name:
                continue
            if name not in florg_A_state:
                continue
            src = florg_A_state[name].to(device=param.device, dtype=param.dtype)
            if tuple(src.shape) != tuple(param.shape):
                raise RuntimeError(
                    f"[florg] shape mismatch while loading {name}: "
                    f"src={tuple(src.shape)}, dst={tuple(param.shape)}"
                )
            param.copy_(src)


def extract_florg_seed_state(model):
    return _clone_seed_state(getattr(model, "_florg_seed_state", None))


def extract_florg_basis_state(model):
    return _clone_basis_state(getattr(model, "_florg_basis_state", None))


def sample_florg_delta_norm(model):
    for layer_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not hasattr(module, "florg_A"):
            continue
        A = module.florg_A.detach().float()
        Q = torch.matmul(A.t(), A)
        delta_w = torch.matmul(module.florg_L.detach().float(), torch.matmul(Q, module.florg_R.detach().float()))
        return layer_name, float(torch.linalg.norm(delta_w).item())
    return None, float("nan")
