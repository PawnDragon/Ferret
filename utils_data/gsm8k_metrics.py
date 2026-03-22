import re

from evaluations import rouge_l_text


_NUM_RE = re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')


def _to_text(value):
    if value is None:
        return ''
    return str(value)


def _extract_first_numeric(text):
    text = _to_text(text)
    match = _NUM_RE.search(text)
    if match is None:
        return None
    return match.group(0)


def _extract_last_numeric(text):
    text = _to_text(text)
    matches = _NUM_RE.findall(text)
    if len(matches) == 0:
        return None
    return matches[-1]


def _canonicalize_numeric(num_text):
    if num_text is None:
        return None
    cleaned = _to_text(num_text).strip().replace(',', '').replace('$', '')
    if cleaned == '':
        return None
    if cleaned.endswith('.'):
        cleaned = cleaned[:-1]
    if cleaned == '':
        return None

    if '/' in cleaned:
        parts = cleaned.split('/')
        if len(parts) == 2:
            left = parts[0].strip()
            right = parts[1].strip()
            try:
                den = float(right)
                if den != 0.0:
                    num = float(left) / den
                    if abs(num - round(num)) < 1e-9:
                        return str(int(round(num)))
                    return f'{num:.12g}'
            except ValueError:
                pass

    try:
        val = float(cleaned)
        if abs(val - round(val)) < 1e-9:
            return str(int(round(val)))
        return f'{val:.12g}'
    except ValueError:
        return cleaned.strip().lower()


def normalize_gsm8k_answer(text):
    normalized = _to_text(text).strip().lower()
    normalized = normalized.replace('####', '').strip()
    normalized = normalized.replace(',', '')
    normalized = normalized.replace('$', '')
    normalized = normalized.rstrip('.').strip()
    normalized = re.sub(r'^(final answer is|the answer is|answer is|final answer|answer)\s*[:：-]?\s*', '', normalized)
    if normalized == '':
        return None
    numeric = _canonicalize_numeric(normalized)
    if numeric is not None:
        return numeric
    return normalized


def extract_gsm8k_gold_final_answer(gold_text):
    raw = _to_text(gold_text)
    if '####' in raw:
        tail = raw.split('####')[-1]
        num = _extract_first_numeric(tail)
        if num is not None:
            return _canonicalize_numeric(num)
        normalized_tail = normalize_gsm8k_answer(tail)
        if normalized_tail is not None and normalized_tail != '':
            return normalized_tail

    num = _extract_last_numeric(raw)
    if num is not None:
        return _canonicalize_numeric(num)
    return normalize_gsm8k_answer(raw)


def extract_gsm8k_pred_final_answer(pred_text):
    raw = _to_text(pred_text)
    lowered = raw.lower()
    patterns = [
        r'final answer is\s*[:：-]?\s*(.+)',
        r'the answer is\s*[:：-]?\s*(.+)',
        r'answer is\s*[:：-]?\s*(.+)',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, lowered))
        if len(matches) > 0:
            tail = raw[matches[-1].start():]
            num = _extract_first_numeric(tail)
            if num is not None:
                return _canonicalize_numeric(num), False
            break

    lines = [line.strip() for line in raw.splitlines() if line.strip() != '']
    if len(lines) > 0:
        last_line = lines[-1]
        num = _extract_last_numeric(last_line)
        if num is not None:
            return _canonicalize_numeric(num), False

    num = _extract_last_numeric(raw)
    if num is not None:
        return _canonicalize_numeric(num), False
    return None, True


def gsm8k_partition_bucket_from_gold(gold_text):
    final = extract_gsm8k_gold_final_answer(gold_text)
    if final is None:
        return 0
    try:
        val = float(final)
        return int(abs(int(round(val)))) % 10
    except ValueError:
        num = _extract_last_numeric(final)
        if num is None:
            return 0
        try:
            val = float(_canonicalize_numeric(num))
            return int(abs(int(round(val)))) % 10
        except ValueError:
            return 0


def compute_gsm8k_metrics(pred_texts, ref_texts):
    if len(pred_texts) != len(ref_texts):
        raise RuntimeError(
            f'[gsm8k] pred/ref size mismatch: '
            f'pred={len(pred_texts)}, ref={len(ref_texts)}'
        )
    n = len(pred_texts)
    if n == 0:
        return {
            'gsm8k_acc': 0.0,
            'gsm8k_rougeL': 0.0,
            'gsm8k_invalid_rate': 0.0,
            'gsm8k_avg_pred_len': 0.0,
            'gsm8k_num_eval_samples': 0,
        }

    n_correct = 0
    n_invalid = 0
    rouge_sum = 0.0
    pred_len_sum = 0.0

    for pred_text, ref_text in zip(pred_texts, ref_texts):
        pred_final, invalid = extract_gsm8k_pred_final_answer(pred_text)
        gold_final = extract_gsm8k_gold_final_answer(ref_text)

        if invalid or pred_final is None:
            n_invalid += 1
        if (pred_final is not None) and (gold_final is not None) and (pred_final == gold_final):
            n_correct += 1

        rouge_sum += float(rouge_l_text(_to_text(pred_text), _to_text(ref_text)))
        pred_len_sum += float(len(_to_text(pred_text).split()))

    n_float = float(max(n, 1))
    return {
        'gsm8k_acc': float(n_correct / n_float),
        'gsm8k_rougeL': float(rouge_sum / n_float),
        'gsm8k_invalid_rate': float(n_invalid / n_float),
        'gsm8k_avg_pred_len': float(pred_len_sum / n_float),
        'gsm8k_num_eval_samples': int(n),
    }
