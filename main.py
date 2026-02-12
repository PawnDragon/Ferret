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
from server import Server
from utils_data.load_data import get_loaders
from optimizers.submuon_utils import estimate_comm_bytes, relative_transport_error

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Algorithm
    parser.add_argument('--algo', type=str, default='ferret', choices=['ferret', 'fedsubmuon'])

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

    # Model
    parser.add_argument('--model', type=str, default='datajuicer/LLaMA-1B-dj-refine-150B')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help=r'learning rate \eta')
    parser.add_argument('--momentum', type=float, default=0.9, help=r'momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay in MeZO')
    parser.add_argument('--grad_clip', type=float, default=-100.0, help='clip the over large loss value, if < 0, disable this feature')

    parser.add_argument('--K', type=int, default=4096, help='ratio of active clients in each round')

    # Ferret server LR schedule
    parser.add_argument('--n_accum', type=int, default=1, help='number of batch for gradient accumulation')
    parser.add_argument('--slr_max', type=float, default=10.0, help='max learning rate for server')
    parser.add_argument('--slr_min', type=float, default=0.0, help='min learning rate for server')
    parser.add_argument('--anneal', type=str, default='linear', help='learning rate annealing for server')

    # FedSubMuon
    parser.add_argument('--rank_r', type=int, default=8)
    parser.add_argument('--beta', type=float, default=0.95)
    parser.add_argument('--ns_steps', type=int, default=5)
    parser.add_argument('--seed_refresh_F', type=int, default=10)

    # Environment
    parser.add_argument('--device', type=int, default=0, help='index of the targeted cuda device')
    parser.add_argument('--log', default=False, action='store_true', help='if `true`, running logs will be recorded in files')
    parser.add_argument('--log_root', default='logs', help='root path of log files')
    parser.add_argument('--seed', default=42, type=int, help='global seed, for reproducibility')
    parser.add_argument('--output_dir', type=str, default='outputs')

    # Evaluation
    parser.add_argument('--eval_metric', default='rouge', type=str, choices=['rouge', 'loss'], help='metric to evaluate global model in the last round')

    # Checkpoints
    parser.add_argument('--save', default=False, action='store_true', help='if `true`, the checkpoint of tuned models will be stored')

    # W&B
    parser.add_argument('--use_wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', type=str, default='ferret')
    parser.add_argument('--wandb_run_name', type=str, default='')

    # Debug
    parser.add_argument('--debug_transport_check', default=False, action='store_true')

    time_stamp = str(time.time())
    args = parser.parse_args()

    if args.algo == 'fedsubmuon':
        args.local_step = 1
        args.batch_or_epoch = 'batch'

    eval_avg_acc = []
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    setup_seed(args.seed)
    list_train_loader, eval_loader, _ = get_loaders(args)

    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir, exist_ok=True)

    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')
    if args.log:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as writer:
            writer.write(config)

    # since only CUDA device is available, load all models on device 0
    args.device = 0

    client_indices_rounds = []
    for _ in range(args.rounds):
        client_indices_rounds.append(np.random.choice(np.arange(args.num_clients), size=int(args.num_clients * args.m), replace=False))

    candidate_seeds = np.random.randint(1, 100000000000, args.K)

    server = Server(args, eval_loader=eval_loader, candidate_seeds=candidate_seeds, log_dir=log_dir)
    client_list = [Client(idx, args, candidate_seeds, list_train_loader[idx]) for idx in range(args.num_clients)]

    if args.debug_transport_check and args.algo == 'fedsubmuon' and len(server.seeds) > 0:
        first_layer = next(iter(server.seeds.keys()))
        out_dim, in_dim = server.submuon_layer_dims[first_layer]
        x_old = torch.randn(args.rank_r, args.rank_r, dtype=torch.float32)
        old_seed = server.seeds[first_layer]
        new_seed = old_seed + 123
        rel_err = relative_transport_error(x_old, old_seed, new_seed, out_dim, in_dim, args.rank_r)
        print(f'[debug] transport relative error @ {first_layer}: {rel_err:.6e}')

    wandb_run = None
    if args.use_wandb and wandb is not None:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name if args.wandb_run_name else None,
            config=vars(args),
        )

    eval_result = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
    eval_avg_acc.append(eval_result)

    if args.algo == 'fedsubmuon':
        server.save_best_submuon_ckpt(eval_result, 0)

    if args.log:
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({'eval_avg_acc': eval_avg_acc}, writer)

    for r in range(1, args.rounds + 1):
        selected_client = [client_list[i] for i in client_indices_rounds[r - 1]]

        if args.algo == 'fedsubmuon':
            server.maybe_refresh_submuon_seeds(r)
            broadcast_state = server.get_submuon_broadcast_state()

            client_payloads = []
            train_losses = []
            for client in selected_client:
                payload = client.local_train_with_seed_pool(server.model, cur_round=r, submuon_state=broadcast_state)
                client_payloads.append(payload)
                train_losses.append(payload['loss'])

            server.aggregate_submuon(client_payloads, selected_client)

            eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
            eval_avg_acc.append(eval_result)
            improved = server.save_best_submuon_ckpt(eval_result, r)

            bytes_up, bytes_down = estimate_comm_bytes(
                num_layers=len(server.seeds),
                rank=args.rank_r,
                num_clients=len(selected_client),
                include_seed=True,
            )
            log_items = {
                'round': r,
                'train/loss_avg': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
                'eval/metric': float(eval_result),
                'comm/bytes_up': float(bytes_up),
                'comm/bytes_down': float(bytes_down),
                'ckpt/improved': int(improved),
            }
            if torch.cuda.is_available():
                log_items['system/mem_alloc'] = float(torch.cuda.memory_allocated())
                log_items['system/mem_reserved'] = float(torch.cuda.memory_reserved())
                log_items['system/max_mem_alloc'] = float(torch.cuda.max_memory_allocated())

            if wandb_run is not None:
                wandb.log(log_items, step=r)

        else:
            for client in selected_client:
                # server.model is pulled after aggregation of the previous round from the server perspective
                # use a global pulling operation to deduplicate the pulling of all clients
                client.local_train_with_seed_pool(deepcopy(server.model), cur_round=r)
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
            eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
            eval_avg_acc.append(eval_result)

        if args.log:
            with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
                json.dump({'eval_avg_acc': eval_avg_acc}, writer)

    # reset seed to have an eval_loader with the same data samples
    args.eval_metric = previous_metric
    setup_seed(args.seed)
    _, eval_loader_final, _ = get_loaders(args, only_eval=True)
    server.eval_loader = eval_loader_final
    eval_result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
    if args.algo == 'fedsubmuon':
        server.save_best_submuon_ckpt(eval_result, args.rounds)

    if args.log:
        with open(os.path.join(log_dir, 'final_eval.json'), 'w') as writer:
            json.dump({f'final_eval_{args.eval_metric}': eval_result}, writer)
    print(f'final round {args.eval_metric}: {eval_result}')

    if wandb_run is not None:
        wandb.finish()
