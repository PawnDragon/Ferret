import torch


LORA_A_TOKEN = '.lora_A.'
LORA_B_TOKEN = '.lora_B.'
_SCALING_LOGGED = False


def _parse_target_modules(raw_target_modules):
    if raw_target_modules is None:
        return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    if isinstance(raw_target_modules, (list, tuple)):
        return [str(item).strip() for item in raw_target_modules if str(item).strip() != '']

    target_text = str(raw_target_modules).strip()
    if target_text == '':
        return ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    if target_text.lower() in ['all-linear', 'all_linear']:
        return 'all-linear'
    if ',' in target_text:
        return [part.strip() for part in target_text.split(',') if part.strip() != '']
    return [target_text]


def _lora_trainable_key(name):
    return (LORA_A_TOKEN in name) or (LORA_B_TOKEN in name)


def _lora_a_key(name):
    return LORA_A_TOKEN in name


def _lora_b_key(name):
    return LORA_B_TOKEN in name


def _classifier_key(name):
    lowered = str(name).lower()
    return ('classifier' in lowered) or lowered.startswith('score.') or ('.score.' in lowered)


def maybe_log_lora_scaling(args):
    global _SCALING_LOGGED
    if _SCALING_LOGGED:
        return
    rank = max(int(getattr(args, 'lora_r', 16)), 1)
    alpha = float(getattr(args, 'lora_alpha', 16.0))
    scaling = alpha / rank
    print(f'[lora] lora_alpha={alpha}, lora_r={rank}, scaling(alpha/r)={scaling:.6f}')
    _SCALING_LOGGED = True


def build_lora_model(backbone_model, args):
    from peft import LoraConfig, TaskType, get_peft_model

    maybe_log_lora_scaling(args)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(getattr(args, 'lora_r', 16)),
        lora_alpha=float(getattr(args, 'lora_alpha', 16.0)),
        lora_dropout=float(getattr(args, 'lora_dropout', 0.0)),
        target_modules=_parse_target_modules(getattr(args, 'lora_target_modules', None)),
        bias=str(getattr(args, 'lora_bias', 'none')),
    )
    lora_model = get_peft_model(backbone_model, lora_config)

    train_classifier = str(getattr(args, 'algo', '')).lower() == 'fedexlora'

    # Enforce backbone freeze for FL LoRA experiments.
    for name, param in lora_model.named_parameters():
        is_trainable = bool(_lora_trainable_key(name))
        if train_classifier and _classifier_key(name):
            is_trainable = True
        param.requires_grad_(is_trainable)

    return lora_model


def extract_lora_state(model):
    lora_state = {}
    for name, param in model.named_parameters():
        if not _lora_trainable_key(name):
            continue
        lora_state[name] = param.detach().cpu().clone()
    return lora_state


def extract_lora_A_state(model):
    lora_a_state = {}
    for name, param in model.named_parameters():
        if not _lora_a_key(name):
            continue
        lora_a_state[name] = param.detach().cpu().clone()
    return lora_a_state


def extract_lora_B_state(model):
    lora_b_state = {}
    for name, param in model.named_parameters():
        if not _lora_b_key(name):
            continue
        lora_b_state[name] = param.detach().cpu().clone()
    return lora_b_state


def extract_classifier_state(model):
    classifier_state = {}
    for name, param in model.named_parameters():
        if not _classifier_key(name):
            continue
        classifier_state[name] = param.detach().cpu().clone()
    return classifier_state


def load_lora_state(model, lora_state):
    if lora_state is None:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _lora_trainable_key(name):
                continue
            if name not in lora_state:
                continue
            src = lora_state[name].to(device=param.device, dtype=param.dtype)
            param.copy_(src)


def load_lora_A_state(model, lora_a_state):
    if lora_a_state is None:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _lora_a_key(name):
                continue
            if name not in lora_a_state:
                continue
            src = lora_a_state[name].to(device=param.device, dtype=param.dtype)
            param.copy_(src)


def load_lora_B_state(model, lora_b_state):
    if lora_b_state is None:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _lora_b_key(name):
                continue
            if name not in lora_b_state:
                continue
            src = lora_b_state[name].to(device=param.device, dtype=param.dtype)
            param.copy_(src)


def load_classifier_state(model, classifier_state):
    if classifier_state is None:
        return
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not _classifier_key(name):
                continue
            if name not in classifier_state:
                continue
            src = classifier_state[name].to(device=param.device, dtype=param.dtype)
            param.copy_(src)


def get_lora_pair_keys(lora_state):
    layer_to_keys = {}
    for key in lora_state.keys():
        if LORA_A_TOKEN in key:
            layer_name = key.split(LORA_A_TOKEN)[0]
            layer_to_keys.setdefault(layer_name, {})['a_key'] = key
        elif LORA_B_TOKEN in key:
            layer_name = key.split(LORA_B_TOKEN)[0]
            layer_to_keys.setdefault(layer_name, {})['b_key'] = key
    return layer_to_keys


def compute_deltaw_from_lora_state(lora_state):
    delta_state = {}
    for layer_name, pair in get_lora_pair_keys(lora_state).items():
        if 'a_key' not in pair or 'b_key' not in pair:
            continue
        # PEFT stores:
        # lora_A.weight: [r, in_features]
        # lora_B.weight: [out_features, r]
        # so deltaW = lora_B @ lora_A with shape [out_features, in_features].
        lora_a = lora_state[pair['a_key']].to(dtype=torch.float32)
        lora_b = lora_state[pair['b_key']].to(dtype=torch.float32)
        delta_state[layer_name] = torch.matmul(lora_b, lora_a).cpu()
    return delta_state


def resolve_layer_name_for_model(layer_name, model):
    modules = dict(model.named_modules())
    if layer_name in modules:
        return layer_name

    parts = layer_name.split('.')
    for start_idx in range(1, len(parts)):
        candidate = '.'.join(parts[start_idx:])
        if candidate in modules:
            return candidate
    return layer_name


def lora_scaling(args):
    rank = max(int(getattr(args, 'lora_r', 16)), 1)
    alpha = float(getattr(args, 'lora_alpha', 16.0))
    return alpha / rank
