import os
import json
from functools import lru_cache

_QWEN3_SELFCHECK_PRINTED = False


def resolve_model_source(model_name_or_path):
    """
    Prefer local filesystem path when it exists; otherwise fallback to
    HuggingFace model id/path string.
    """
    local_path = os.path.expanduser(model_name_or_path)
    if os.path.exists(local_path):
        return local_path
    return model_name_or_path


@lru_cache(maxsize=128)
def _is_qwen3_model_cached(name: str) -> bool:
    s = (name or '').lower()
    if ('qwen3' in s) or ('qwen/qwen3' in s):
        return True

    local_path = os.path.expanduser(name or '')
    config_path = os.path.join(local_path, 'config.json')
    if not os.path.isfile(config_path):
        return False

    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
    except Exception:
        return False

    model_type = str(cfg.get('model_type', '')).lower()
    if 'qwen3' in model_type:
        return True

    name_or_path = str(cfg.get('_name_or_path', '')).lower()
    if ('qwen3' in name_or_path) or ('qwen/qwen3' in name_or_path):
        return True

    architectures = cfg.get('architectures', [])
    if isinstance(architectures, list):
        arch_text = ' '.join(str(x).lower() for x in architectures)
        if 'qwen3' in arch_text:
            return True

    return False


def is_qwen3_model(name: str) -> bool:
    return _is_qwen3_model_cached(name or '')


def format_chat_text(tokenizer, messages, add_generation_prompt, model_name_or_path, enable_thinking=False):
    if not is_qwen3_model(model_name_or_path):
        return None
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def maybe_print_qwen3_selfcheck(tokenizer, model_name_or_path):
    global _QWEN3_SELFCHECK_PRINTED
    if _QWEN3_SELFCHECK_PRINTED or (not is_qwen3_model(model_name_or_path)):
        return
    _QWEN3_SELFCHECK_PRINTED = True

    print('Detected model: Qwen3')
    print(f'tokenizer.pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}')

    dummy_messages = [{'role': 'user', 'content': 'Please introduce yourself in one sentence.'}]
    try:
        preview = tokenizer.apply_chat_template(
            dummy_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        print(f'Qwen3 chat template preview: {preview[:200]}')
    except Exception as exc:
        print(f'[warn] Qwen3 chat template self-check failed: {exc}')
