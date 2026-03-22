import json
import os

import pandas as pd


GSM8K_INSTRUCTION = "Solve the following grade-school math problem step by step."


def _load_jsonl_rows(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as reader:
        for line_idx, line in enumerate(reader):
            content = line.strip()
            if content == '':
                continue
            try:
                rows.append(json.loads(content))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f'[gsm8k] failed to parse jsonl at {path}:{line_idx + 1}: {exc}'
                ) from exc
    return rows


def _load_json_rows(path):
    with open(path, 'r', encoding='utf-8') as reader:
        payload = json.load(reader)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if 'data' in payload and isinstance(payload['data'], list):
            return payload['data']
        raise RuntimeError(
            f'[gsm8k] json payload at {path} must be a list or contain list field `data`'
        )
    raise RuntimeError(f'[gsm8k] unsupported json payload type at {path}: {type(payload)}')


def _load_parquet_rows(path):
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f'[gsm8k] failed to load parquet file: {path} ({exc})') from exc
    return frame.to_dict(orient='records')


def _normalize_record(row, split_name, row_idx):
    if not isinstance(row, dict):
        raise RuntimeError(
            f'[gsm8k] malformed {split_name} sample at index={row_idx}: expected dict, got {type(row)}'
        )
    if 'question' not in row:
        raise RuntimeError(
            f'[gsm8k] missing required field `question` in {split_name} sample index={row_idx}, keys={list(row.keys())[:20]}'
        )
    if 'answer' not in row:
        raise RuntimeError(
            f'[gsm8k] missing required field `answer` in {split_name} sample index={row_idx}, keys={list(row.keys())[:20]}'
        )

    question = '' if row['question'] is None else str(row['question'])
    answer = '' if row['answer'] is None else str(row['answer'])
    if question.strip() == '':
        raise RuntimeError(f'[gsm8k] empty question in {split_name} sample index={row_idx}')
    if answer.strip() == '':
        raise RuntimeError(f'[gsm8k] empty answer in {split_name} sample index={row_idx}')

    return {
        'instruction': GSM8K_INSTRUCTION,
        'input': question,
        'output': answer,
    }


def _resolve_format_paths(gsm8k_root):
    root = os.path.expanduser(str(gsm8k_root))
    if (not os.path.exists(root)) or (not os.path.isdir(root)):
        raise FileNotFoundError(f'[gsm8k] dataset root does not exist or is not a directory: {root}')

    parquet_train = os.path.join(root, 'train-00000-of-00001.parquet')
    parquet_test = os.path.join(root, 'test-00000-of-00001.parquet')
    jsonl_train = os.path.join(root, 'train.jsonl')
    jsonl_test = os.path.join(root, 'test.jsonl')
    json_train = os.path.join(root, 'train.json')
    json_test = os.path.join(root, 'test.json')

    if os.path.exists(parquet_train) and os.path.exists(parquet_test):
        return root, 'parquet', {'train': parquet_train, 'test': parquet_test}
    if os.path.exists(parquet_train) != os.path.exists(parquet_test):
        missing = parquet_test if os.path.exists(parquet_train) else parquet_train
        raise FileNotFoundError(f'[gsm8k] missing required split file: {missing}')

    if os.path.exists(jsonl_train) and os.path.exists(jsonl_test):
        return root, 'jsonl', {'train': jsonl_train, 'test': jsonl_test}
    if os.path.exists(jsonl_train) != os.path.exists(jsonl_test):
        missing = jsonl_test if os.path.exists(jsonl_train) else jsonl_train
        raise FileNotFoundError(f'[gsm8k] missing required split file: {missing}')

    if os.path.exists(json_train) and os.path.exists(json_test):
        return root, 'json', {'train': json_train, 'test': json_test}
    if os.path.exists(json_train) != os.path.exists(json_test):
        missing = json_test if os.path.exists(json_train) else json_train
        raise FileNotFoundError(f'[gsm8k] missing required split file: {missing}')

    raise FileNotFoundError(
        '[gsm8k] could not find local train/test files under root. Expected one of: '
        '`train-00000-of-00001.parquet` + `test-00000-of-00001.parquet` (preferred), '
        '`train.jsonl` + `test.jsonl`, or `train.json` + `test.json`.'
    )


def load_gsm8k_local_splits(gsm8k_root):
    root, fmt, split_paths = _resolve_format_paths(gsm8k_root)

    if fmt == 'parquet':
        train_rows = _load_parquet_rows(split_paths['train'])
        test_rows = _load_parquet_rows(split_paths['test'])
    elif fmt == 'jsonl':
        train_rows = _load_jsonl_rows(split_paths['train'])
        test_rows = _load_jsonl_rows(split_paths['test'])
    else:
        train_rows = _load_json_rows(split_paths['train'])
        test_rows = _load_json_rows(split_paths['test'])

    train_examples = [_normalize_record(row, 'train', idx) for idx, row in enumerate(train_rows)]
    test_examples = [_normalize_record(row, 'test', idx) for idx, row in enumerate(test_rows)]

    if len(train_examples) == 0:
        raise RuntimeError(f'[gsm8k] train split is empty under root: {root}')
    if len(test_examples) == 0:
        raise RuntimeError(f'[gsm8k] test split is empty under root: {root}')

    return train_examples, test_examples
