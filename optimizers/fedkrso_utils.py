import hashlib
import math

import torch


def _stable_seed(seed, layer_name):
    layer_text = str(layer_name)
    digest = hashlib.sha256(layer_text.encode("utf-8")).digest()
    layer_hash = int.from_bytes(digest[:8], byteorder="little", signed=False)
    mixed = (int(seed) ^ layer_hash) % (2**63 - 1)
    return int(mixed)


def make_krso_projection(seed, layer_name, rank, in_dim, device, dtype):
    rank_eff = int(min(max(int(rank), 1), int(in_dim)))
    if rank_eff <= 0:
        raise RuntimeError(
            f"[fedkrso] invalid projection rank for layer={layer_name}: rank={rank}, in_dim={in_dim}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(seed, layer_name))
    proj = torch.randn(rank_eff, int(in_dim), generator=generator, dtype=torch.float32)
    proj.mul_(1.0 / math.sqrt(float(rank_eff)))
    return proj.to(device=device, dtype=dtype).contiguous()
