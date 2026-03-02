import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from copy import deepcopy

from client import Client
from evaluate_dolly import run_evaluate_from_checkpoint
from server import Server
from optimizers.submuon_utils import relative_transport_error
from utils_data.comm_utils import compute_comm_size
from utils_data.load_data import get_loaders
from utils_data.rebase_controller import AdaptiveRebaseController

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import wandb
except ImportError:
    wandb = None


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cosine_annealing_lr(r, R, eta_max, eta_min=0):
    return eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * r / R)) / 2


def linear_annealing_lr(r, R, eta_max, eta_min=0):
    return eta_min + (eta_max - eta_min) * (1 - r / R)


def is_finite_scalar(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def maybe_save_best_ckpt(server, args, metric, cur_round, log_dir):
    if args.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
        return server.save_best_submuon_ckpt(metric, cur_round)
    if args.algo in ['fedit', 'flora', 'fedsalora', 'fedexlora']:
        return server.save_best_lora_ckpt(metric, cur_round)
    if args.algo == 'fedavg':
        return server.save_best_fedavg_ckpt(metric, cur_round)

    improved = (metric < server.best_metric) if args.eval_metric == 'loss' else (metric > server.best_metric)
    if not improved:
        return False

    server.best_metric = metric
    ckpt_path = os.path.join(log_dir, 'best.pt')
    torch.save(
        {
            'algo': args.algo,
            'backbone_state_dict': server.model.state_dict(),
            'round': int(cur_round),
            'best_metric': float(metric),
            'hparams': {
                'algo': args.algo,
                'lr': float(args.lr),
            },
        },
        ckpt_path,
    )
    print(f'[ckpt] saved to: {ckpt_path}')
    return True


def apply_early_stop_state(early_state, val_loss, cur_round, min_delta):
    if not is_finite_scalar(val_loss):
        return False
    if (early_state['best_val_loss'] - float(val_loss)) > float(min_delta):
        early_state['best_val_loss'] = float(val_loss)
        early_state['best_round'] = int(cur_round)
        early_state['no_improve_count'] = 0
        return True
    early_state['no_improve_count'] += 1
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Algorithm
    parser.add_argument('--algo', type=lambda x: x.lower(), default='ferret', choices=['ferret', 'fedsubmuon', 'fedsubadam', 'fedsubsgd', 'fedit', 'flora', 'fedsalora', 'fedexlora', 'fedavg'])

    # Federation
    parser.add_argument('--num_clients', type=int, default=200, help='N in our paper')
    parser.add_argument('-m', type=float, default=0.05, help='ratio of activate clients in each round')
    parser.add_argument('--rounds', type=int, default=40, help='the total number of rounds')
    parser.add_argument('--local_step', type=int, default=200, help=r'$\tau in our paper')
    parser.add_argument('--batch_or_epoch', type=str, default='batch', choices=['epoch', 'batch'])
    parser.add_argument('--equal_weight', default=False, action='store_true', help='if `true`, the weights among clients for aggregation are the same')

    # Data
    parser.add_argument('--dataset', type=str, default='instruct', choices=['instruct', 'dolly'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size > 1 may cause error during running')
    parser.add_argument('--max_length', type=int, default=1024, help='the max number of tokens of a data instance')
    parser.add_argument('--use_prompts', default=True, help='if `true`, the prompt template from alpaca is adopted')

    # Dolly-only
    parser.add_argument('--iid', type=str, default='dir0.5', help=r'`dir{alpha}` means that \alpha in Dirichlet distribution, `0` means IID split')
    parser.add_argument('--zerotask', default=7, type=int, help='the index of the task for evaluation in dolly-15K')
    parser.add_argument('--dataset_subsample', type=float, default=1.0, help='used for sampling a subset from the original dataset, only effective for dolly-15K')
    parser.add_argument('--ni_root', type=str, default='./data/NI', help='root directory for Natural Instructions dataset')

    # Model
    parser.add_argument('--model', type=str, default='datajuicer/LLaMA-1B-dj-refine-150B')
    parser.add_argument('--model_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'], help='model dtype for loading/training/eval')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help=r'learning rate \eta')
    parser.add_argument('--momentum', type=float, default=0.9, help=r'momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay in MeZO')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='beta1 for AdamW in FedAvg/FedIT')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 for AdamW in FedAvg/FedIT')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon for AdamW in FedAvg/FedIT')
    parser.add_argument('--grad_clip', type=float, default=-100.0, help='clip the over large loss value, if < 0, disable this feature')
    parser.add_argument('--max_grad_norm', type=float, default=-1.0, help='gradient clipping threshold for FedAvg; <=0 disables clipping')

    parser.add_argument('--K', type=int, default=4096, help='ratio of active clients in each round')

    # Ferret server LR schedule
    parser.add_argument('--n_accum', type=int, default=1, help='number of batch for gradient accumulation')
    parser.add_argument('--slr_max', type=float, default=10.0, help='max learning rate for server')
    parser.add_argument('--slr_min', type=float, default=0.0, help='min learning rate for server')
    parser.add_argument('--anneal', type=str, default='linear', help='learning rate annealing for server')

    # FedSubMuon
    parser.add_argument('--rank_r', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--ns_steps', type=int, default=5)
    parser.add_argument('--seed_refresh_F', type=int, default=10)
    parser.add_argument('--stop_F', type=int, default=-1, help='stop seed refresh from this round onward; <=0 disables stopping')
    parser.add_argument('--adaptive_rebase', default=False, action='store_true', help='enable adaptive restart-style rebase for FedSubMuon')
    parser.add_argument('--rebase_patience', type=int, default=5, help='no-improve rounds before triggering rebase')
    parser.add_argument('--rebase_cooldown', type=int, default=3, help='cooldown rounds after each rebase')
    parser.add_argument('--max_rebase', type=int, default=2, help='maximum rebase count')
    parser.add_argument('--rebase_min_delta', type=float, default=1e-4, help='minimum loss improvement threshold for rebase controller')

    # LoRA (FedIT / FLoRA)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_target_modules', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')
    parser.add_argument('--lora_bias', type=str, default='none', choices=['none', 'all', 'lora_only'])

    # Environment
    parser.add_argument('--device', type=int, default=0, help='index of the targeted cuda device')
    parser.add_argument('--log', default=False, action='store_true', help='if `true`, running logs will be recorded in files')
    parser.add_argument('--log_root', default='logs', help='root path of log files')
    parser.add_argument('--seed', default=42, type=int, help='global seed, for reproducibility')
    # Backward-compatible only; checkpoints are now saved under log_dir.
    parser.add_argument('--output_dir', type=str, default='outputs')

    # Evaluation
    parser.add_argument('--eval_metric', default='rouge', type=str, choices=['rouge', 'loss'], help='metric to evaluate global model in the last round')
    parser.add_argument('--round_eval_false', default=False, action='store_true', help='if true, skip evaluation during training rounds')
    parser.add_argument('--early_stop', default=False, action='store_true', help='if true, stop training early when eval metric does not improve')
    parser.add_argument('--early_stop_patience', type=int, default=8, help='number of rounds without significant improvement before stopping')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4, help='minimum significant improvement in eval metric')
    parser.add_argument('--early_stop_metric', type=str, default='eval/loss', help='metric name used by early stopping (currently supports eval/loss)')

    # Checkpoints
    parser.add_argument('--save', default=False, action='store_true', help='if `true`, keep checkpoint files after run; otherwise remove them at run end')

    # W&B
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', type=str, default='ferret')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--wandb_run_name', type=str, default='')

    # Debug
    parser.add_argument('--debug_transport_check', default=False, action='store_true')
    parser.add_argument('--debug_nan', default=False, action='store_true', help='enable NaN/Inf diagnostics during local training')
    parser.add_argument('--debug_nan_first_steps', type=int, default=1, help='print debug snapshots for first N local steps on the target client')
    parser.add_argument('--debug_nan_client_idx', type=int, default=0, help='only print NaN debug for this client index; <0 means all clients')
    parser.add_argument('--debug_nan_skip_optim', default=False, action='store_true', help='skip optimizer step when non-finite loss/grad is detected')

    time_stamp = str(time.time())
    args = parser.parse_args()

    eval_avg_acc = []
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'
    early_stop_active = bool(args.early_stop)
    adaptive_rebase_active = bool(args.algo == 'fedsubmuon' and args.adaptive_rebase)
    if early_stop_active and args.round_eval_false:
        print('[warn] --early_stop requires round evaluation; disable early stopping because --round_eval_false is set')
        early_stop_active = False
    if early_stop_active and args.early_stop_metric != 'eval/loss':
        print(f'[warn] unsupported --early_stop_metric={args.early_stop_metric}; fallback to eval/loss')
    if args.adaptive_rebase and args.algo != 'fedsubmuon':
        print(f'[warn] --adaptive_rebase is only used by fedsubmuon, but current algo={args.algo}; ignore adaptive rebase')
        adaptive_rebase_active = False
    if adaptive_rebase_active and args.round_eval_false:
        print('[warn] --adaptive_rebase requires round evaluation; disable adaptive rebase because --round_eval_false is set')
        adaptive_rebase_active = False
    if adaptive_rebase_active and int(args.early_stop_patience) <= int(args.rebase_patience):
        print(
            f'[warn] recommended early_stop_patience({args.early_stop_patience}) > '
            f'rebase_patience({args.rebase_patience})'
        )
    control_min_delta = float(getattr(args, 'early_stop_min_delta', getattr(args, 'rebase_min_delta', 1e-4)))
    user_keep_ckpt = bool(args.save)
    args.keep_ckpt = bool(user_keep_ckpt)
    if not user_keep_ckpt:
        print('[info] checkpoints will be saved during training for eval/early-stop, then removed at run end because --save is false')
    # Always enable checkpoint writes during training.
    args.save = True

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    requested_device = int(args.device)
    cuda_count = torch.cuda.device_count()
    if cuda_count > 0:
        if requested_device < 0 or requested_device >= cuda_count:
            print(f'[warn] --device {requested_device} is invalid for {cuda_count} visible GPU(s); fallback to 0')
            requested_device = 0
        os.environ['CUDA_VISIBLE_DEVICES'] = str(requested_device)
    else:
        print('[warn] No CUDA GPU visible before training startup; running in CPU fallback mode')

    setup_seed(args.seed)

    wandb_run = None
    if args.use_wandb and wandb is None:
        print('[warn] --use_wandb is set but wandb package is not installed; disable wandb logging.')
        args.use_wandb = False
    if args.use_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity if args.wandb_entity != '' else None),
            name=(args.wandb_name or args.wandb_run_name or None),
            config={
                'lr': float(args.lr),
                'rank_r': int(args.rank_r),
            },
        )
        if 'lr' in wandb.config:
            args.lr = float(wandb.config.lr)
        if 'rank_r' in wandb.config:
            args.rank_r = int(wandb.config.rank_r)

    list_train_loader, eval_loader, _ = get_loaders(args)

    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir, exist_ok=True)
    args.ckpt_dir = log_dir

    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')
    if args.log:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as writer:
            writer.write(config)

    # After CUDA_VISIBLE_DEVICES pinning, use local cuda:0 in process.
    args.device = 0

    client_indices_rounds = []
    for _ in range(args.rounds):
        client_indices_rounds.append(np.random.choice(np.arange(args.num_clients), size=int(args.num_clients * args.m), replace=False))

    candidate_seeds = np.random.randint(1, 100000000000, args.K)

    server = Server(args, eval_loader=eval_loader, candidate_seeds=candidate_seeds, log_dir=log_dir)
    client_list = [Client(idx, args, candidate_seeds, list_train_loader[idx]) for idx in range(args.num_clients)]

    if args.debug_transport_check and args.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd'] and len(server.seeds) > 0:
        first_layer = next(iter(server.seeds.keys()))
        out_dim, in_dim = server.submuon_layer_dims[first_layer]
        x_old = torch.randn(args.rank_r, args.rank_r, dtype=torch.float32)
        old_seed = server.seeds[first_layer]
        new_seed = old_seed + 123
        rel_err = relative_transport_error(x_old, old_seed, new_seed, out_dim, in_dim, args.rank_r)
        print(f'[debug] transport relative error @ {first_layer}: {rel_err:.6e}')

    eval_result = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
    eval_avg_acc.append(eval_result)
    if wandb_run is not None:
        init_log = {
            'round': 0,
            'train/loss_avg': float('nan'),
            'train/wall_clock_time': float('nan'),
            'train/peak_gpu_mem': float('nan'),
            'train/comm_up_bytes': float('nan'),
            'train/comm_down_bytes': float('nan'),
            'eval/loss': float(eval_result),
            'ctrl/no_improve': 0,
            'ctrl/rebase_count': 0,
            'ctrl/cooldown_left': 0,
            'ctrl/did_rebase': 0,
        }
        if torch.cuda.is_available():
            init_log['system/mem_alloc'] = float(torch.cuda.memory_allocated())
            init_log['system/mem_reserved'] = float(torch.cuda.memory_reserved())
            init_log['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())
        wandb.log(init_log, step=0)

    maybe_save_best_ckpt(
        server=server,
        args=args,
        metric=eval_result,
        cur_round=0,
        log_dir=log_dir,
    )

    early_state = {
        'best_val_loss': float(eval_result) if is_finite_scalar(eval_result) else float('inf'),
        'best_round': 0,
        'no_improve_count': 0,
    }
    rebase_controller = None
    if adaptive_rebase_active:
        rebase_controller = AdaptiveRebaseController(
            best_loss=early_state['best_val_loss'],
            best_round=early_state['best_round'],
            min_delta=control_min_delta,
            rebase_patience=int(args.rebase_patience),
            rebase_cooldown=int(args.rebase_cooldown),
            max_rebase=int(args.max_rebase),
            enable_rebase=True,
            enable_early_stop=early_stop_active,
            early_stop_patience=int(args.early_stop_patience),
        )
    last_round = 0
    early_stopped = False

    if args.log:
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({'eval_avg_acc': eval_avg_acc}, writer)

    for r in range(1, args.rounds + 1):
        selected_client = [client_list[i] for i in client_indices_rounds[r - 1]]
        train_losses = []
        total_comm_up_bytes = 0
        total_comm_down_bytes = 0
        round_start_time = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        if args.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            if not (adaptive_rebase_active and args.algo == 'fedsubmuon'):
                server.maybe_refresh_submuon_seeds(r)
            broadcast_state = server.get_submuon_broadcast_state()
            total_comm_down_bytes = compute_comm_size(server.get_broadcast_state()) * len(selected_client)

            client_payloads = []
            for client in selected_client:
                payload = client.local_train_with_seed_pool(server.model, cur_round=r, submuon_state=broadcast_state)
                client_payloads.append(payload)
                train_losses.append(payload['loss'])
                total_comm_up_bytes += compute_comm_size({k: v for k, v in payload.items() if k in ['x', 'm', 'v']})

            server.aggregate_submuon(client_payloads, selected_client)
            wall_clock = time.time() - round_start_time
            peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0

            if args.round_eval_false:
                eval_result = float('nan')
                improved = 0
            else:
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                eval_avg_acc.append(eval_result)
                if adaptive_rebase_active and args.algo == 'fedsubmuon':
                    improved = 0
                else:
                    improved = int(
                        maybe_save_best_ckpt(
                            server=server,
                            args=args,
                            metric=eval_result,
                            cur_round=r,
                            log_dir=log_dir,
                        )
                    )

            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'train/wall_clock_time': float(wall_clock),
                'train/peak_gpu_mem': float(peak_gpu_mem),
                'train/comm_up_bytes': float(total_comm_up_bytes),
                'train/comm_down_bytes': float(total_comm_down_bytes),
                'comm/bytes_up': float(total_comm_up_bytes),
                'comm/bytes_down': float(total_comm_down_bytes),
                'eval/loss': float(eval_result),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

        elif args.algo in ['fedit', 'flora']:
            broadcast_lora = server.get_fedit_broadcast_state() if args.algo == 'fedit' else None
            broadcast_named_optim_state = server.get_fedit_broadcast_optim_state() if args.algo == 'fedit' else None
            total_comm_down_bytes = compute_comm_size(server.get_broadcast_state()) * len(selected_client)
            client_payloads = []
            for client in selected_client:
                payload = client.local_train_with_seed_pool(
                    deepcopy(server.model),
                    cur_round=r,
                    lora_state=broadcast_lora,
                    global_named_optim_state=broadcast_named_optim_state,
                )
                client_payloads.append(payload)
                train_losses.append(payload['loss'])
                if args.algo == 'fedit':
                    total_comm_up_bytes += compute_comm_size(
                        {
                            'lora_state': payload.get('lora_state', {}),
                            'named_optim_state': payload.get('named_optim_state', None),
                        }
                    )
                else:
                    total_comm_up_bytes += compute_comm_size(payload.get('lora_state', {}))

            server.aggregate_lora(client_payloads, selected_client)
            wall_clock = time.time() - round_start_time
            peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            if args.round_eval_false:
                eval_result = float('nan')
                improved = 0
            else:
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                eval_avg_acc.append(eval_result)
                improved = int(
                    maybe_save_best_ckpt(
                        server=server,
                        args=args,
                        metric=eval_result,
                        cur_round=r,
                        log_dir=log_dir,
                    )
                )

            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'train/wall_clock_time': float(wall_clock),
                'train/peak_gpu_mem': float(peak_gpu_mem),
                'train/comm_up_bytes': float(total_comm_up_bytes),
                'train/comm_down_bytes': float(total_comm_down_bytes),
                'comm/bytes_up': float(total_comm_up_bytes),
                'comm/bytes_down': float(total_comm_down_bytes),
                'eval/loss': float(eval_result),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

        elif args.algo == 'fedsalora':
            broadcast_lora_A = server.get_fedsalora_broadcast_state()
            total_comm_down_bytes = compute_comm_size(server.get_broadcast_state()) * len(selected_client)
            client_a_states = []
            for client in selected_client:
                payload = client.local_train_with_seed_pool(
                    deepcopy(server.model),
                    cur_round=r,
                    lora_A_state=broadcast_lora_A,
                )
                client_a_states.append(payload.get('lora_A_state', {}))
                train_losses.append(payload['loss'])
                total_comm_up_bytes += compute_comm_size(payload.get('lora_A_state', {}))

            server.aggregate_fedsalora(client_a_states, selected_client)
            wall_clock = time.time() - round_start_time
            peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            if args.round_eval_false:
                eval_result = float('nan')
                improved = 0
            else:
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                eval_avg_acc.append(eval_result)
                improved = int(
                    maybe_save_best_ckpt(
                        server=server,
                        args=args,
                        metric=eval_result,
                        cur_round=r,
                        log_dir=log_dir,
                    )
                )

            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'train/wall_clock_time': float(wall_clock),
                'train/peak_gpu_mem': float(peak_gpu_mem),
                'train/comm_up_bytes': float(total_comm_up_bytes),
                'train/comm_down_bytes': float(total_comm_down_bytes),
                'comm/bytes_up': float(total_comm_up_bytes),
                'comm/bytes_down': float(total_comm_down_bytes),
                'eval/loss': float(eval_result),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

        elif args.algo == 'fedexlora':
            broadcast_state = server.get_broadcast_state_fedexlora()
            total_comm_down_bytes = compute_comm_size(broadcast_state) * len(selected_client)
            broadcast_lora_A = broadcast_state.get('global_lora_A_state', {})
            broadcast_lora_B = broadcast_state.get('global_lora_B_state', {})
            broadcast_classifier = broadcast_state.get('global_classifier_state', {})
            client_payloads = []
            for client in selected_client:
                payload = client.local_train_with_seed_pool(
                    deepcopy(server.model),
                    cur_round=r,
                    lora_A_state=broadcast_lora_A,
                    lora_B_state=broadcast_lora_B,
                    classifier_state=broadcast_classifier,
                )
                client_payloads.append(payload)
                train_losses.append(payload['loss'])
                total_comm_up_bytes += compute_comm_size(
                    {
                        'lora_A_state': payload.get('lora_A_state', {}),
                        'lora_B_state': payload.get('lora_B_state', {}),
                        'classifier_state': payload.get('classifier_state', {}),
                    }
                )

            server.aggregate_fedexlora(client_payloads, selected_client, cur_round=r)
            wall_clock = time.time() - round_start_time
            peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            if args.round_eval_false:
                eval_result = float('nan')
                improved = 0
            else:
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                eval_avg_acc.append(eval_result)
                improved = int(
                    maybe_save_best_ckpt(
                        server=server,
                        args=args,
                        metric=eval_result,
                        cur_round=r,
                        log_dir=log_dir,
                    )
                )

            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'train/wall_clock_time': float(wall_clock),
                'train/peak_gpu_mem': float(peak_gpu_mem),
                'train/comm_up_bytes': float(total_comm_up_bytes),
                'train/comm_down_bytes': float(total_comm_down_bytes),
                'comm/bytes_up': float(total_comm_up_bytes),
                'comm/bytes_down': float(total_comm_down_bytes),
                'eval/loss': float(eval_result),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

        elif args.algo == 'fedavg':
            broadcast_state = server.get_fedavg_broadcast_state()
            broadcast_named_optim_state = server.get_fedavg_broadcast_optim_state()
            total_comm_down_bytes = compute_comm_size(
                {
                    'backbone_state_dict': broadcast_state,
                    'global_named_optim_state': broadcast_named_optim_state,
                }
            ) * len(selected_client)
            server.begin_fedavg_aggregation(selected_client)
            for client_idx, client in enumerate(selected_client):
                payload = client.local_train_with_seed_pool(
                    server.model,
                    cur_round=r,
                    fedavg_global_state=broadcast_state,
                    global_named_optim_state=broadcast_named_optim_state,
                )
                train_losses.append(payload['loss'])
                total_comm_up_bytes += compute_comm_size(
                    {
                        'model_state_dict': payload.get('model_state_dict', {}),
                        'named_optim_state': payload.get('named_optim_state', None),
                    }
                )
                server.accumulate_fedavg_payload(payload, client_idx)

            server.finalize_fedavg_aggregation()
            wall_clock = time.time() - round_start_time
            peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            if args.round_eval_false:
                eval_result = float('nan')
                improved = 0
            else:
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                eval_avg_acc.append(eval_result)
                improved = int(
                    maybe_save_best_ckpt(
                        server=server,
                        args=args,
                        metric=eval_result,
                        cur_round=r,
                        log_dir=log_dir,
                    )
                )

            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'train/wall_clock_time': float(wall_clock),
                'train/peak_gpu_mem': float(peak_gpu_mem),
                'train/comm_up_bytes': float(total_comm_up_bytes),
                'train/comm_down_bytes': float(total_comm_down_bytes),
                'comm/bytes_up': float(total_comm_up_bytes),
                'comm/bytes_down': float(total_comm_down_bytes),
                'eval/loss': float(eval_result),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

        else:
            total_comm_down_bytes = compute_comm_size(server.get_broadcast_state()) * len(selected_client)
            for client in selected_client:
                # server.model is pulled after aggregation of the previous round from the server perspective
                # use a global pulling operation to deduplicate the pulling of all clients
                loss_val = client.local_train_with_seed_pool(deepcopy(server.model), cur_round=r)
                if loss_val is not None:
                    train_losses.append(float(loss_val))
                total_comm_up_bytes += compute_comm_size(getattr(client, 'local_seed_pool', {}))
            server.aggregate_seed_pool(selected_client, cur_round=r)

            if args.anneal == 'linear':
                args.slr = linear_annealing_lr(r - 1, args.rounds, args.slr_max, args.slr_min)
            elif args.anneal == 'cosine':
                args.slr = cosine_annealing_lr(r - 1, args.rounds, args.slr_max, args.slr_min)
            elif args.anneal == 'no':
                args.slr = args.slr_max
            print(f'round: {r}, server learning rate: {args.slr}')

            # server gets the latest global model from the accumulated scalar gradients
            server.update_global_model_by_seed_pool()
            wall_clock = time.time() - round_start_time
            peak_gpu_mem = float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            if args.round_eval_false:
                eval_result = float('nan')
                improved = 0
            else:
                eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
                eval_avg_acc.append(eval_result)
                improved = int(
                    maybe_save_best_ckpt(
                        server=server,
                        args=args,
                        metric=eval_result,
                        cur_round=r,
                        log_dir=log_dir,
                    )
                )
            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'train/wall_clock_time': float(wall_clock),
                'train/peak_gpu_mem': float(peak_gpu_mem),
                'train/comm_up_bytes': float(total_comm_up_bytes),
                'train/comm_down_bytes': float(total_comm_down_bytes),
                'comm/bytes_up': float(total_comm_up_bytes),
                'comm/bytes_down': float(total_comm_down_bytes),
                'eval/loss': float(eval_result),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

        pending_adaptive_early_stop = False
        did_rebase = 0
        if adaptive_rebase_active and args.algo == 'fedsubmuon' and rebase_controller is not None:
            if (not args.round_eval_false) and is_finite_scalar(eval_result):
                ctrl_result = rebase_controller.step(cur_round=r, curr_loss=float(eval_result))
                if ctrl_result['improved']:
                    improved = int(
                        maybe_save_best_ckpt(
                            server=server,
                            args=args,
                            metric=eval_result,
                            cur_round=r,
                            log_dir=log_dir,
                        )
                    )
                    log_items['ckpt/improved'] = int(improved)
                if ctrl_result['did_rebase']:
                    did_rebase = int(bool(server.trigger_rebase(cur_round=r)))
                    print(
                        f'Round {r}: REBASE triggered (count={ctrl_result["rebase_count"]}), '
                        f'best_loss={ctrl_result["best_loss"]:.6f}, curr_loss={float(eval_result):.6f}, '
                        f'cooldown={int(args.rebase_cooldown)}'
                    )
                pending_adaptive_early_stop = bool(ctrl_result['should_early_stop'])

            ctrl_snapshot = rebase_controller.snapshot()
            early_state['best_val_loss'] = float(ctrl_snapshot['best_loss'])
            early_state['best_round'] = int(ctrl_snapshot['best_round'])
            early_state['no_improve_count'] = int(ctrl_snapshot['no_improve'])
            ctrl_no_improve = int(ctrl_snapshot['no_improve'])
            ctrl_rebase_count = int(ctrl_snapshot['rebase_count'])
            ctrl_cooldown_left = int(ctrl_snapshot['cooldown_left'])
        else:
            ctrl_no_improve = int(early_state['no_improve_count'])
            ctrl_rebase_count = 0
            ctrl_cooldown_left = 0

        log_items['ctrl/no_improve'] = int(ctrl_no_improve)
        log_items['ctrl/rebase_count'] = int(ctrl_rebase_count)
        log_items['ctrl/cooldown_left'] = int(ctrl_cooldown_left)
        log_items['ctrl/did_rebase'] = int(did_rebase)

        print(
            f'Round {r}: wall_clock_time = {log_items["train/wall_clock_time"]:.2f} s, '
            f'peak_gpu_mem = {log_items["train/peak_gpu_mem"]:.2f} MB, '
            f'comm_up_bytes = {int(log_items["train/comm_up_bytes"])}, '
            f'comm_down_bytes = {int(log_items["train/comm_down_bytes"])}, '
            f'eval/loss = {log_items["eval/loss"]:.6f}'
        )

        if wandb_run is not None:
            wandb.log(log_items, step=r)

        if args.log:
            with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
                json.dump({'eval_avg_acc': eval_avg_acc}, writer)

        last_round = r
        if adaptive_rebase_active and args.algo == 'fedsubmuon':
            if pending_adaptive_early_stop:
                early_stopped = True
                print(
                    '[early-stop] triggered at round '
                    f"{r} | best_round={early_state['best_round']} "
                    f"| best_loss={early_state['best_val_loss']:.6f} "
                    f"| no_improve_count={early_state['no_improve_count']} "
                    f"| rebase_count={rebase_controller.rebase_count if rebase_controller is not None else 0}"
                )
                break
            continue

        if (not args.round_eval_false) and is_finite_scalar(eval_result):
            # Keep best-round / best-loss tracking always-on for final report consistency.
            apply_early_stop_state(
                early_state=early_state,
                val_loss=eval_result,
                cur_round=r,
                min_delta=args.early_stop_min_delta,
            )
            if early_stop_active and early_state['no_improve_count'] >= int(args.early_stop_patience):
                early_stopped = True
                print(
                    '[early-stop] triggered at round '
                    f"{r} | best_round={early_state['best_round']} "
                    f"| best_loss={early_state['best_val_loss']:.6f} "
                    f"| no_improve_count={early_state['no_improve_count']}"
                )
                break

    best_ckpt_path = os.path.join(log_dir, 'best.pt')
    final_round_idx = int(last_round if last_round > 0 else args.rounds)
    if not os.path.exists(best_ckpt_path):
        print(f'[warn] best checkpoint not found at {best_ckpt_path}; skip final eval')
    else:
        data_args = {
            'dataset': args.dataset,
            'zerotask': args.zerotask,
            'dataset_subsample': args.dataset_subsample,
            'iid': args.iid,
            'num_clients': args.num_clients,
            'batch_size': args.batch_size,
            'max_length': args.max_length,
            'use_prompts': args.use_prompts,
            'ni_root': args.ni_root,
        }
        eval_args = {
            'eval_metric': previous_metric,
            'algo': 'auto',
            'seed': args.seed,
            'rank_r': args.rank_r,
            'lr': args.lr,
            'beta': args.beta,
            'ns_steps': args.ns_steps,
            'weight_decay': args.weight_decay,
            'model_dtype': args.model_dtype,
            'n_accum': args.n_accum,
            'grad_clip': args.grad_clip,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'lora_target_modules': args.lora_target_modules,
            'lora_bias': args.lora_bias,
        }
        final_eval = run_evaluate_from_checkpoint(
            checkpoint_path=best_ckpt_path,
            model_name_or_path=args.model,
            tokenizer_name_or_path=args.model,
            data_args=data_args,
            eval_args=eval_args,
            device=args.device,
        )
        final_metric_name = previous_metric
        final_metric_val = float(final_eval['result'])
        if args.log:
            with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
                json.dump(
                    {
                        f'final_eval_{final_metric_name}': final_metric_val,
                        'checkpoint': best_ckpt_path,
                        'round_end': final_round_idx,
                        'early_stopped': bool(early_stopped),
                        'early_stop_best_round': int(early_state['best_round']),
                        'early_stop_best_loss': float(early_state['best_val_loss']),
                    },
                    writer,
                )
        print(f'final round {final_metric_name} (best ckpt): {final_metric_val}')
        if wandb_run is not None:
            wandb.log(
                {
                    f'final/{final_metric_name}': final_metric_val,
                    'final/eval_samples': int(final_eval.get('eval_samples', 0)),
                    'final/used_best_checkpoint': 1,
                    'final/round_end': int(final_round_idx),
                    'final/early_stopped': int(bool(early_stopped)),
                    'final/early_stop_best_round': int(early_state['best_round']),
                    'final/early_stop_best_loss': float(early_state['best_val_loss']),
                },
                step=final_round_idx,
            )

    if not user_keep_ckpt:
        removed_ckpt = 0
        if os.path.isdir(log_dir):
            for file_name in os.listdir(log_dir):
                file_path = os.path.join(log_dir, file_name)
                if (not os.path.isfile(file_path)):
                    continue
                lower_name = file_name.lower()
                if lower_name.endswith('.pt') or lower_name.endswith('.bin'):
                    os.remove(file_path)
                    removed_ckpt += 1
        print(f'[ckpt] removed {removed_ckpt} checkpoint file(s) because --save is false')

    if wandb_run is not None:
        wandb.finish()
