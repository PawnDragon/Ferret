import torch


def compute_comm_size(state):
    total_bytes = 0

    if isinstance(state, torch.Tensor):
        return int(state.nelement() * state.element_size())

    if isinstance(state, dict):
        for value in state.values():
            total_bytes += compute_comm_size(value)
        return int(total_bytes)

    if isinstance(state, (list, tuple)):
        for value in state:
            total_bytes += compute_comm_size(value)
        return int(total_bytes)

    # Keep accounting for non-tensor protocol fields such as seeds.
    if isinstance(state, (int, float, bool)):
        return 8

    return 0
