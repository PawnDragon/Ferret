import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from utils_data.math_metrics import extract_math_gold_final_answer


MATH_INSTRUCTION = (
    "Solve the following competition math problem. "
    "Show your reasoning clearly and put the final answer in \\boxed{}."
)


def _load_parquet_rows(path):
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f'[math] failed to load parquet file: {path} ({exc})') from exc
    return frame.to_dict(orient='records')


def _parse_level(raw_level, split_name, row_idx):
    if isinstance(raw_level, (int, np.integer)):
        return int(raw_level)
    if isinstance(raw_level, float):
        if np.isnan(raw_level):
            return None
        return int(raw_level)
    level_text = '' if raw_level is None else str(raw_level).strip()
    if level_text == '':
        return None
    match = re.search(r'(-?\d+)', level_text)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_record(row, split_name, row_idx):
    if not isinstance(row, dict):
        raise RuntimeError(
            f'[math] malformed {split_name} sample at index={row_idx}: expected dict, got {type(row)}'
        )
    required_fields = ['problem', 'solution', 'extracted_solution', 'type', 'level']
    for field_name in required_fields:
        if field_name not in row:
            raise RuntimeError(
                f'[math] missing required field `{field_name}` in {split_name} sample index={row_idx}, '
                f'keys={list(row.keys())[:20]}'
            )

    problem = '' if row['problem'] is None else str(row['problem'])
    solution = '' if row['solution'] is None else str(row['solution'])
    extracted_solution = '' if row['extracted_solution'] is None else str(row['extracted_solution'])
    subject = '' if row['type'] is None else str(row['type'])
    if problem.strip() == '':
        return None, 'empty_problem'
    if solution.strip() == '':
        return None, 'empty_solution'
    if subject.strip() == '':
        return None, 'empty_type'

    level = _parse_level(row['level'], split_name=split_name, row_idx=row_idx)
    if level is None:
        return None, 'invalid_level'
    row_id = row.get('id', None)
    if row_id is None:
        row_id = f'{split_name}-{row_idx}'
    final_answer = extract_math_gold_final_answer(extracted_solution)
    if final_answer is None:
        return None, 'unparseable_extracted_solution'

    return {
        'id': str(row_id),
        'instruction': MATH_INSTRUCTION,
        'input': problem,
        'output': solution,
        'final_answer': final_answer,
        'subject': subject.strip(),
        'level': int(level),
        'dataset_name': 'math',
        'split': str(split_name),
    }, None


def _resolve_math_paths(dataset_path):
    root = os.path.expanduser(str(dataset_path))
    if (not os.path.exists(root)) or (not os.path.isdir(root)):
        raise FileNotFoundError(f'[math] dataset root does not exist or is not a directory: {root}')

    train_path = os.path.join(root, 'train-00000-of-00001.parquet')
    test_path = os.path.join(root, 'test-00000-of-00001.parquet')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f'[math] missing required split file: {train_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'[math] missing required split file: {test_path}')
    return root, train_path, test_path


def _split_train_dev_stratified(train_examples, dev_size, seed):
    total = len(train_examples)
    if total <= 1:
        return [dict(item) for item in train_examples], []

    target_dev = int(max(min(int(dev_size), total - 1), 0))
    if target_dev == 0:
        return [dict(item) for item in train_examples], []

    buckets = defaultdict(list)
    for idx, item in enumerate(train_examples):
        key = (str(item.get('subject', 'unknown')), int(item.get('level', -1)))
        buckets[key].append(int(idx))

    rng = np.random.RandomState(int(seed))
    sorted_keys = sorted(buckets.keys())
    shuffled_by_key = {}
    for key in sorted_keys:
        arr = np.array(buckets[key], dtype=np.int64)
        rng.shuffle(arr)
        shuffled_by_key[key] = arr.tolist()

    base_alloc = {}
    frac_alloc = []
    allocated = 0
    for key in sorted_keys:
        group_size = len(shuffled_by_key[key])
        raw = float(group_size * target_dev) / float(total)
        whole = int(np.floor(raw))
        whole = min(whole, group_size)
        base_alloc[key] = whole
        allocated += whole
        frac_alloc.append((raw - whole, key))

    remain = int(target_dev - allocated)
    if remain > 0:
        frac_alloc.sort(key=lambda item: (-item[0], item[1][0], item[1][1]))
        for _, key in frac_alloc:
            if remain <= 0:
                break
            if base_alloc[key] < len(shuffled_by_key[key]):
                base_alloc[key] += 1
                remain -= 1

    dev_indices = set()
    for key in sorted_keys:
        take_n = int(base_alloc[key])
        if take_n <= 0:
            continue
        chosen = shuffled_by_key[key][:take_n]
        for idx in chosen:
            dev_indices.add(int(idx))

    train_out = []
    dev_out = []
    for idx, item in enumerate(train_examples):
        copied = dict(item)
        if idx in dev_indices:
            copied['split'] = 'dev'
            dev_out.append(copied)
        else:
            copied['split'] = 'train'
            train_out.append(copied)
    return train_out, dev_out


def load_math_local_splits(dataset_path, seed=42, dev_size=500):
    root, train_path, test_path = _resolve_math_paths(dataset_path)
    train_rows = _load_parquet_rows(train_path)
    test_rows = _load_parquet_rows(test_path)

    train_examples = []
    test_examples = []
    train_skipped = 0
    test_skipped = 0
    train_skip_reasons = defaultdict(int)
    test_skip_reasons = defaultdict(int)
    for idx, row in enumerate(train_rows):
        normalized, skip_reason = _normalize_record(row, 'train', idx)
        if normalized is None:
            train_skipped += 1
            train_skip_reasons[str(skip_reason or 'unknown')] += 1
            continue
        train_examples.append(normalized)
    for idx, row in enumerate(test_rows):
        normalized, skip_reason = _normalize_record(row, 'test', idx)
        if normalized is None:
            test_skipped += 1
            test_skip_reasons[str(skip_reason or 'unknown')] += 1
            continue
        test_examples.append(normalized)
    if len(train_examples) == 0:
        raise RuntimeError(f'[math] train split is empty under root: {root}')
    if len(test_examples) == 0:
        raise RuntimeError(f'[math] test split is empty under root: {root}')

    train_split, dev_split = _split_train_dev_stratified(
        train_examples=train_examples,
        dev_size=int(dev_size),
        seed=int(seed),
    )
    if len(train_split) == 0:
        raise RuntimeError('[math] train split became empty after dev split; check dataset size and dev policy')
    if (train_skipped + test_skipped) > 0:
        print(
            f'[math] skipped malformed samples: '
            f'train={train_skipped} reasons={dict(sorted(train_skip_reasons.items()))}, '
            f'test={test_skipped} reasons={dict(sorted(test_skip_reasons.items()))}'
        )
    return train_split, dev_split, test_examples
