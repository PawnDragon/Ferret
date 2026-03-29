import re
from collections import defaultdict
from fractions import Fraction

from evaluations import rouge_l_text


_SIMPLE_DECIMAL_RE = re.compile(r'^[+-]?\d+(?:\.\d+)?$')
_SIMPLE_FRACTION_RE = re.compile(r'^[+-]?\d+\s*/\s*[+-]?\d+$')
_LATEX_FRAC_RE = re.compile(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}')


def _to_text(value):
    if value is None:
        return ''
    return str(value)


def _extract_last_boxed_content(text):
    raw = _to_text(text)
    token = '\\boxed{'
    pos = 0
    boxed_values = []
    while True:
        start = raw.find(token, pos)
        if start < 0:
            break
        i = start + len(token)
        depth = 1
        while i < len(raw):
            ch = raw[i]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    boxed_values.append(raw[start + len(token):i])
                    pos = i + 1
                    break
            i += 1
        else:
            break
    if len(boxed_values) == 0:
        return None
    return boxed_values[-1]


def _strip_outer_math_delimiters(text):
    value = _to_text(text).strip()
    while value.startswith('$') and value.endswith('$') and len(value) >= 2:
        value = value[1:-1].strip()
    return value


def _strip_outer_boxed(text):
    value = _to_text(text).strip()
    while True:
        if not value.startswith('\\boxed{'):
            break
        inner = _extract_last_boxed_content(value)
        if inner is None:
            break
        rebuilt = f'\\boxed{{{inner}}}'
        if rebuilt != value:
            break
        value = inner.strip()
    return value


def _replace_simple_latex_frac(text):
    def _repl(match):
        left = match.group(1).strip()
        right = match.group(2).strip()
        return f'{left}/{right}'

    return _LATEX_FRAC_RE.sub(_repl, _to_text(text))


def _canonicalize_numeric(text):
    value = _to_text(text).strip()
    if _SIMPLE_FRACTION_RE.match(value):
        left, right = value.split('/', 1)
        try:
            frac = Fraction(int(left.strip()), int(right.strip()))
            num_val = float(frac)
            if abs(num_val - round(num_val)) < 1e-9:
                return str(int(round(num_val)))
            return f'{num_val:.12g}'
        except (ValueError, ZeroDivisionError):
            return value
    if _SIMPLE_DECIMAL_RE.match(value):
        try:
            num_val = float(value)
            if abs(num_val - round(num_val)) < 1e-9:
                return str(int(round(num_val)))
            return f'{num_val:.12g}'
        except ValueError:
            return value
    return value


def normalize_math_answer(text):
    value = _to_text(text).strip()
    if value == '':
        return None
    value = _strip_outer_boxed(value)
    value = _strip_outer_math_delimiters(value)
    value = _replace_simple_latex_frac(value)
    value = value.replace('\\left', '').replace('\\right', '')
    value = value.replace(',', '')
    value = value.rstrip('.').strip()
    value = re.sub(
        r'^(final answer is|the answer is|answer is|final answer|answer)\s*[:：-]?\s*',
        '',
        value,
        flags=re.IGNORECASE,
    ).strip()
    if value == '':
        return None
    numeric = _canonicalize_numeric(value)
    return numeric.strip().lower()


def extract_math_gold_final_answer(extracted_solution, fallback_solution=None):
    normalized = normalize_math_answer(extracted_solution)
    if normalized is not None:
        return normalized
    if fallback_solution is None:
        return None
    inferred, invalid = extract_math_pred_final_answer(fallback_solution)
    if invalid:
        return None
    return inferred


def extract_math_pred_final_answer(pred_text):
    boxed = _extract_last_boxed_content(pred_text)
    if boxed is not None:
        normalized = normalize_math_answer(boxed)
        if normalized is not None:
            return normalized, False

    patterns = [
        r'final answer is\s*[:：-]?\s*(.+)',
        r'the answer is\s*[:：-]?\s*(.+)',
        r'answer is\s*[:：-]?\s*(.+)',
    ]
    raw = _to_text(pred_text)
    for pattern in patterns:
        matches = list(re.finditer(pattern, raw, flags=re.IGNORECASE))
        if len(matches) == 0:
            continue
        tail = matches[-1].group(1).strip()
        if '\n' in tail:
            tail = tail.split('\n', 1)[0].strip()
        normalized = normalize_math_answer(tail)
        if normalized is not None:
            return normalized, False
    return None, True


def compute_math_metrics(pred_texts, ref_texts, gold_finals=None, subjects=None, levels=None):
    if len(pred_texts) != len(ref_texts):
        raise RuntimeError(
            f'[math] pred/ref size mismatch: pred={len(pred_texts)}, ref={len(ref_texts)}'
        )
    n = len(pred_texts)
    if n == 0:
        return {
            'math_acc': 0.0,
            'math_rougeL': 0.0,
            'math_invalid_rate': 0.0,
            'math_avg_pred_len': 0.0,
            'math_num_eval_samples': 0,
            'math_acc_by_subject': {},
            'math_acc_by_level': {},
        }

    if gold_finals is None:
        gold_finals = [None for _ in range(n)]
    if subjects is None:
        subjects = ['unknown' for _ in range(n)]
    if levels is None:
        levels = ['unknown' for _ in range(n)]
    if len(gold_finals) != n:
        raise RuntimeError(
            f'[math] gold_finals size mismatch: gold={len(gold_finals)}, n={n}'
        )

    n_correct = 0
    n_invalid = 0
    rouge_sum = 0.0
    pred_len_sum = 0.0

    by_subject_total = defaultdict(int)
    by_subject_correct = defaultdict(int)
    by_level_total = defaultdict(int)
    by_level_correct = defaultdict(int)

    for idx in range(n):
        pred_text = _to_text(pred_texts[idx])
        ref_text = _to_text(ref_texts[idx])
        subject = _to_text(subjects[idx]).strip() or 'unknown'
        level = _to_text(levels[idx]).strip() or 'unknown'

        pred_final, invalid = extract_math_pred_final_answer(pred_text)
        gold_final = extract_math_gold_final_answer(
            gold_finals[idx],
            fallback_solution=ref_text,
        )

        if invalid or pred_final is None:
            n_invalid += 1
        is_correct = bool(
            (pred_final is not None)
            and (gold_final is not None)
            and (pred_final == gold_final)
        )
        if is_correct:
            n_correct += 1

        by_subject_total[subject] += 1
        by_subject_correct[subject] += int(is_correct)
        by_level_total[level] += 1
        by_level_correct[level] += int(is_correct)

        rouge_sum += float(rouge_l_text(pred_text, ref_text))
        pred_len_sum += float(len(pred_text.split()))

    n_float = float(max(n, 1))
    acc_by_subject = {
        str(key): float(by_subject_correct[key] / max(by_subject_total[key], 1))
        for key in sorted(by_subject_total.keys())
    }
    acc_by_level = {
        str(key): float(by_level_correct[key] / max(by_level_total[key], 1))
        for key in sorted(by_level_total.keys())
    }
    return {
        'math_acc': float(n_correct / n_float),
        'math_rougeL': float(rouge_sum / n_float),
        'math_invalid_rate': float(n_invalid / n_float),
        'math_avg_pred_len': float(pred_len_sum / n_float),
        'math_num_eval_samples': int(n),
        'math_acc_by_subject': acc_by_subject,
        'math_acc_by_level': acc_by_level,
    }
