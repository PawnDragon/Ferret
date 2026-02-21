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
from optimizers.lora_utils import build_lora_model, load_lora_state
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
    v_global = None
    seeds = None
    saved_algo = None
    global_lora_state = None
    global_deltaW_state = None
    lora_hparams = {}

    if isinstance(payload, dict) and 'backbone_state_dict' in payload:
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        x_global = payload.get('x_global', None)
        m_global = payload.get('m_global', None)
        v_global = payload.get('v_global', None)
        seeds = payload.get('seeds', None)
        global_lora_state = payload.get('global_lora_state', None)
        global_deltaW_state = payload.get('global_deltaW_state', None)
        lora_hparams = payload.get('lora_hparams', {}) if isinstance(payload.get('lora_hparams', {}), dict) else {}
        if isinstance(payload.get('hparams', {}), dict):
            saved_algo = payload['hparams'].get('algo', None)
        if saved_algo is None:
            saved_algo = payload.get('algo', None)

        if saved_algo in ['fedit', 'flora'] or global_lora_state is not None or global_deltaW_state is not None:
            ckpt_type = 'lora_best'
        else:
            ckpt_type = 'fedsubmuon_best'
    elif isinstance(payload, dict):
        model.load_state_dict(payload, strict=True)
    else:
        raise ValueError(f'Unsupported checkpoint format: {type(payload)}')

    return ckpt_type, x_global, m_global, v_global, seeds, saved_algo, global_lora_state, global_deltaW_state, lora_hparams


def to_left_padded_inputs(input_ids, attention_mask, pad_token_id):
    """
    Convert right-padded batch tensors to left-padded layout for decoder-only generation.
    """
    bs, seq_len = input_ids.shape
    left_input_ids = torch.full_like(input_ids, pad_token_id)
    left_attention_mask = torch.zeros_like(attention_mask)
    for i in range(bs):
        valid_tokens = input_ids[i][attention_mask[i].bool()]
        n = valid_tokens.numel()
        if n > 0:
            left_input_ids[i, seq_len - n:] = valid_tokens
            left_attention_mask[i, seq_len - n:] = 1
    return left_input_ids, left_attention_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--algo', type=str, default='auto', choices=['auto', 'ferret', 'fedsubmuon', 'fedsubadam', 'fedsubsgd', 'fedit', 'flora'])

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
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    parser.add_argument('--lora_bias', type=str, default='none')

    parser.add_argument('--save_json', type=str, default='')

    args = parser.parse_args()

    device, _ = resolve_runtime_device(args.device)
    setup_seed(args.seed)

    # Keep exactly the same final-eval path as main.py.
    setup_seed(args.seed)
    _, eval_loader, tokenizer = get_loaders(args, only_eval=True)
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
    tokenizer.padding_side = 'left'

    ckpt_type = 'none'
    x_global = None
    m_global = None
    v_global = None
    seeds = None
    saved_algo = None
    global_lora_state = None
    global_deltaW_state = None
    lora_hparams = {}
    if args.checkpoint:
        ckpt_type, x_global, m_global, v_global, seeds, saved_algo, global_lora_state, global_deltaW_state, lora_hparams = load_checkpoint_into_model(model, args.checkpoint)
        print(f'[info] Loaded checkpoint: {args.checkpoint} ({ckpt_type})')

    eval_algo = args.algo
    if eval_algo == 'auto':
        if saved_algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd', 'fedit', 'flora']:
            eval_algo = saved_algo
        else:
            if global_lora_state is not None:
                eval_algo = 'fedit'
            elif global_deltaW_state is not None:
                eval_algo = 'flora'
            else:
                eval_algo = 'fedsubmuon' if (x_global is not None and seeds is not None) else 'ferret'

    # Align LoRA eval settings with saved checkpoint when available.
    if isinstance(lora_hparams, dict) and len(lora_hparams) > 0:
        if 'lora_r' in lora_hparams:
            args.lora_r = int(lora_hparams['lora_r'])
        if 'lora_alpha' in lora_hparams:
            args.lora_alpha = float(lora_hparams['lora_alpha'])
        if 'lora_dropout' in lora_hparams:
            args.lora_dropout = float(lora_hparams['lora_dropout'])
        if 'lora_target_modules' in lora_hparams:
            args.lora_target_modules = lora_hparams['lora_target_modules']
        if 'lora_bias' in lora_hparams:
            args.lora_bias = lora_hparams['lora_bias']

    model = model.to(device)
    model.eval()
    eval_model = model

    framework = None
    if eval_algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
        if x_global is None or seeds is None:
            raise ValueError('FedSub eval requires checkpoint with x_global and seeds')
        if eval_algo in ['fedsubmuon', 'fedsubadam'] and m_global is None:
            raise ValueError('FedSubMuon/FedSubAdam eval requires checkpoint with m_global')
        if len(x_global) == 0:
            raise ValueError('FedSub checkpoint contains empty x_global')
        first_key = next(iter(x_global.keys()))
        ckpt_rank = int(x_global[first_key].shape[0])
        if int(args.rank_r) != ckpt_rank:
            print(f'[info] rank_r mismatch (args={args.rank_r}, ckpt={ckpt_rank}); override args.rank_r with ckpt rank')
            args.rank_r = ckpt_rank
        args.algo = eval_algo
        framework = FerretFramework(model, args=args, lr=args.lr, candidate_seeds=[])
        framework.set_submuon_state(
            x_global,
            m_global if eval_algo in ['fedsubmuon', 'fedsubadam'] else None,
            seeds,
            trainable=False,
            v_state=v_global if eval_algo == 'fedsubadam' else None,
        )
    elif eval_algo == 'fedit':
        if global_lora_state is None:
            raise ValueError('FedIT eval requires checkpoint with global_lora_state')
        eval_model = build_lora_model(model, args)
        load_lora_state(eval_model, global_lora_state)
        eval_model = eval_model.to(device)
        eval_model.eval()
    elif eval_algo == 'flora':
        # Flora checkpoints are evaluated from the saved backbone state.
        # Current training already applies global delta to backbone each round.
        eval_model = model

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
                outputs = eval_model(**batch)
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
                bs = input_ids.size(0)
                for i in range(bs):
                    valid_input = input_ids[i][attention_mask[i].bool()].unsqueeze(0)
                    valid_mask = torch.ones_like(valid_input, device=device)
                    output_ids = eval_model.generate(
                        input_ids=valid_input,
                        attention_mask=valid_mask,
                        pad_token_id=tokenizer.pad_token_id,
                        max_new_tokens=128,
                        num_beams=1,
                    )
                    prompt_len = int(valid_mask[0].sum().item())
                    pred_ids = output_ids[0][prompt_len:]
                    ref_ids = label_ids[i]
                    if ref_ids.numel() > 0:
                        ref_ids = ref_ids[ref_ids >= 0]
                    metric_total += rouge_score(pred_ids, ref_ids, tokenizer)
                pbar.update(1)
                num_eval += bs
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
