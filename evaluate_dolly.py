import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from evaluations import rouge_score
from optimizers.ferret_optimizer import FerretFramework
from utils_data.load_data import get_loaders
from utils_data.model_loader import resolve_model_source


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_runtime_device(requested_device):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    cuda_count = torch.cuda.device_count()
    if cuda_count <= 0:
        print('[warn] No CUDA GPU visible, using CPU')
        return torch.device('cpu'), 'cpu'

    dev = int(requested_device)
    if dev < 0 or dev >= cuda_count:
        print(f'[warn] --device {dev} is invalid for {cuda_count} visible GPU(s); fallback to 0')
        dev = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(dev)
    return torch.device('cuda:0'), 'cuda'


def load_checkpoint_into_model(model, ckpt_path):
    payload = torch.load(ckpt_path, map_location='cpu')

    ckpt_type = 'state_dict'
    x_global = None
    m_global = None
    seeds = None

    if isinstance(payload, dict) and 'backbone_state_dict' in payload:
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        x_global = payload.get('x_global', None)
        m_global = payload.get('m_global', None)
        seeds = payload.get('seeds', None)
        ckpt_type = 'fedsubmuon_best'
    elif isinstance(payload, dict):
        model.load_state_dict(payload, strict=True)
    else:
        raise ValueError(f'Unsupported checkpoint format: {type(payload)}')

    return ckpt_type, x_global, m_global, seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--algo', type=str, default='auto', choices=['auto', 'ferret', 'fedsubmuon'])

    # Data/eval args to keep dolly processing aligned with main.py
    parser.add_argument('--dataset', type=str, default='dolly', choices=['dolly'])
    parser.add_argument('--zerotask', type=int, default=7)
    parser.add_argument('--dataset_subsample', type=float, default=1.0)
    parser.add_argument('--iid', type=str, default='dir0.5')
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--use_prompts', default=True)
    parser.add_argument('--eval_metric', type=str, default='loss', choices=['loss', 'rouge'])

    # Runtime/model args used by framework/server-style behavior
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rank_r', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--ns_steps', type=int, default=5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--n_accum', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=-100.0)

    parser.add_argument('--save_json', type=str, default='')

    args = parser.parse_args()

    device, _ = resolve_runtime_device(args.device)
    setup_seed(args.seed)

    # Keep exactly the same dolly eval split logic as main.py via get_loaders.
    _, eval_loader, tokenizer = get_loaders(args, only_eval=False)
    print(f'[info] Dolly zerotask eval | zerotask={args.zerotask}, eval_samples={len(eval_loader.dataset)}')

    model_source = resolve_model_source(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        device_map='cpu',
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    ckpt_type = 'none'
    x_global = None
    m_global = None
    seeds = None
    if args.checkpoint:
        ckpt_type, x_global, m_global, seeds = load_checkpoint_into_model(model, args.checkpoint)
        print(f'[info] Loaded checkpoint: {args.checkpoint} ({ckpt_type})')

    eval_algo = args.algo
    if eval_algo == 'auto':
        eval_algo = 'fedsubmuon' if (x_global is not None and seeds is not None) else 'ferret'

    model = model.to(device)
    model.eval()

    framework = None
    if eval_algo == 'fedsubmuon':
        if x_global is None or m_global is None or seeds is None:
            raise ValueError('FedSubMuon eval requires checkpoint with x_global, m_global, seeds')
        args.algo = 'fedsubmuon'
        framework = FerretFramework(model, args=args, lr=args.lr, candidate_seeds=[])
        framework.set_submuon_state(x_global, m_global, seeds, trainable=False)

    result = None
    if args.eval_metric == 'loss':
        loss_total = 0.0
        num_eval = 0
        pbar = tqdm(range(len(eval_loader)))
        with torch.inference_mode():
            for batch in eval_loader:
                batch = {
                    'input_ids': batch['input_ids'].to(device),
                    'labels': batch['labels'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                }
                outputs = model(**batch)
                loss = outputs.loss
                pbar.update(1)
                if torch.isnan(loss):
                    continue
                loss_total += loss
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                pbar.set_description(f'eval loss: {loss_total / num_eval}')
        result = float((loss_total / num_eval).item())
        print(f'[result] eval_loss={result}')
    else:
        metric_total = 0.0
        num_eval = 0
        pbar = tqdm(range(len(eval_loader)))
        with torch.inference_mode():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                label_ids = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                )
                metric_total += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], tokenizer)
                pbar.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                pbar.set_description(f'eval rouge: {metric_total / num_eval}')
        result = float(metric_total / num_eval)
        print(f'[result] eval_rouge={result}')

    if framework is not None:
        framework.clear_submuon_state()

    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(
                {
                    'model': args.model,
                    'checkpoint': args.checkpoint,
                    'algo': eval_algo,
                    'eval_metric': args.eval_metric,
                    'zerotask': args.zerotask,
                    'result': result,
                    'eval_samples': len(eval_loader.dataset),
                },
                f,
                indent=2,
            )
        print(f'[info] Saved eval json to {args.save_json}')


if __name__ == '__main__':
    main()
