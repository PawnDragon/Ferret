import argparse
import json
from dataclasses import dataclass

from transformers import AutoTokenizer

from utils_data.default_tokens import DefaultToken
from utils_data.llm_dataset import PROMPT_DICT
from utils_data.model_loader import format_chat_text, is_qwen3_model, resolve_model_source


@dataclass
class Sample:
    instruction: str
    context: str
    response: str
    category: str


def load_dolly_sample(jsonl_path: str, sample_index: int) -> Sample:
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    if sample_index < 0 or sample_index >= len(lines):
        raise IndexError(f'sample_index out of range: {sample_index}, total={len(lines)}')
    item = json.loads(lines[sample_index])
    return Sample(
        instruction=item.get('instruction', ''),
        context=item.get('context', ''),
        response=item.get('response', ''),
        category=item.get('category', ''),
    )


def build_qwen3_user_content(instruction: str, input_text: str) -> str:
    instruction = '' if instruction is None else str(instruction)
    input_text = '' if input_text is None else str(input_text)
    if input_text != '':
        return f'{instruction}\n\nInput:\n{input_text}'
    return instruction


def prepare_tokenizer(model_name_or_path: str, max_length: int):
    model_source = resolve_model_source(model_name_or_path)
    qwen3 = is_qwen3_model(model_source)
    kwargs = {'use_fast': True}
    if qwen3:
        kwargs['trust_remote_code'] = True
    tokenizer = AutoTokenizer.from_pretrained(model_source, **kwargs)
    tokenizer.model_max_length = max_length

    special_tokens = {}
    if tokenizer.pad_token is None and (not qwen3):
        special_tokens['pad_token'] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens['eos_token'] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens['bos_token'] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens['unk_token'] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, qwen3, model_source


def build_source_and_target(sample: Sample, tokenizer, qwen3: bool, model_source: str):
    if qwen3:
        messages = [
            {
                'role': 'user',
                'content': build_qwen3_user_content(sample.instruction, sample.context),
            }
        ]
        source = format_chat_text(
            tokenizer=tokenizer,
            messages=messages,
            add_generation_prompt=True,
            model_name_or_path=model_source,
            enable_thinking=False,
        )
    else:
        if sample.context != '':
            source = PROMPT_DICT['prompt_input'].format(
                instruction=sample.instruction,
                input=sample.context,
            )
        else:
            source = PROMPT_DICT['prompt_no_input'].format(instruction=sample.instruction)
    target = f'{sample.response}{tokenizer.eos_token}'
    return source, target


def tokenize_like_training(source: str, target: str, tokenizer):
    example = source + target
    tok_example = tokenizer(
        example,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=tokenizer.model_max_length,
    )
    tok_source = tokenizer(
        source,
        return_tensors='pt',
        padding='longest',
        truncation=True,
        max_length=tokenizer.model_max_length,
    )

    input_ids = tok_example.input_ids[0]
    attention_mask = tok_example.attention_mask[0]
    labels = input_ids.clone()
    source_len = int(tok_source.input_ids[0].ne(tokenizer.pad_token_id).sum().item())
    labels[:source_len] = DefaultToken.IGNORE_INDEX.value
    valid_mask = labels.ne(DefaultToken.IGNORE_INDEX.value)
    valid_mask = valid_mask & attention_mask.bool()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'source_len': source_len,
        'example_len': int(attention_mask.sum().item()),
        'loss_token_count': int(valid_mask.sum().item()),
        'labels_valid_text': tokenizer.decode(labels[valid_mask], skip_special_tokens=False),
        'boundary_window': tokenizer.decode(
            input_ids[max(0, source_len - 20):min(input_ids.numel(), source_len + 20)],
            skip_special_tokens=False,
        ),
        'prefix_decode_at_source_len': tokenizer.decode(input_ids[:source_len], skip_special_tokens=False),
    }


def analyze_model(model_name_or_path: str, sample: Sample, max_length: int):
    tokenizer, qwen3, model_source = prepare_tokenizer(model_name_or_path, max_length=max_length)
    source, target = build_source_and_target(sample, tokenizer, qwen3=qwen3, model_source=model_source)
    token_stats = tokenize_like_training(source, target, tokenizer)
    return {
        'model': model_name_or_path,
        'resolved_model_source': model_source,
        'is_qwen3': qwen3,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'padding_side': tokenizer.padding_side,
        'raw_sample': sample,
        'source': source,
        'target': target,
        **token_stats,
    }


def print_report(tag: str, r: dict, show_chars: int):
    print(f'===== {tag} =====')
    print(f'model: {r["model"]}')
    print(f'resolved_model_source: {r["resolved_model_source"]}')
    print(f'is_qwen3: {r["is_qwen3"]}')
    print(f'pad_token_id: {r["pad_token_id"]}, eos_token_id: {r["eos_token_id"]}, padding_side: {r["padding_side"]}')
    print(f'input_ids_len(attended): {r["example_len"]}')
    print(f'labels_valid_tokens(loss_participating): {r["loss_token_count"]}')
    print()
    print('--- raw sample ---')
    print(f'instruction: {r["raw_sample"].instruction}')
    print(f'context: {r["raw_sample"].context}')
    print(f'response: {r["raw_sample"].response}')
    print()
    print('--- source (prompt/messages rendered) ---')
    print(r['source'][:show_chars])
    print()
    print('--- target ---')
    print(r['target'][:show_chars])
    print()
    print('--- labels valid decoded text (should be answer side) ---')
    print(r['labels_valid_text'][:show_chars])
    print()
    print('--- token boundary window around source_len ---')
    print(r['boundary_window'][:show_chars])
    print()
    print('--- decoded prefix at source_len ---')
    print(r['prefix_decode_at_source_len'][:show_chars])
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--llama_model', type=str, required=True)
    parser.add_argument('--qwen_model', type=str, required=True)
    parser.add_argument('--dolly_jsonl', type=str, default='data/databricks-dolly-15k.jsonl')
    parser.add_argument('--sample_index', type=int, default=0)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--show_chars', type=int, default=800)
    args = parser.parse_args()

    sample = load_dolly_sample(args.dolly_jsonl, args.sample_index)
    llama_result = analyze_model(args.llama_model, sample, max_length=args.max_length)
    qwen_result = analyze_model(args.qwen_model, sample, max_length=args.max_length)

    print_report('LLAMA', llama_result, args.show_chars)
    print_report('QWEN', qwen_result, args.show_chars)

    print('===== COMPARISON =====')
    print(f'loss tokens | llama={llama_result["loss_token_count"]} | qwen={qwen_result["loss_token_count"]}')
    print(f'attended len | llama={llama_result["example_len"]} | qwen={qwen_result["example_len"]}')
    print(f'source len   | llama={llama_result["source_len"]} | qwen={qwen_result["source_len"]}')


if __name__ == '__main__':
    main()
