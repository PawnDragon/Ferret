import os
import math
import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from evaluations import *
from optimizers.ferret_optimizer import FerretFramework
from optimizers.submuon_utils import init_submuon_state, transport_state
from utils_data.default_tokens import DefaultToken
from utils_data.model_loader import resolve_model_source


def softmax(vec):
    vec = vec - np.max(vec)
    exp_x = np.exp(vec)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def min_max_norm(vec):
    min_val = np.min(vec)
    return (vec - min_val) / (np.max(vec) + 1e-10 - min_val)


class Server(object):
    def __init__(self, args, eval_loader, candidate_seeds, log_dir):
        self.args = args
        self.eval_loader = eval_loader
        self.candidate_seeds = candidate_seeds
        model_source = resolve_model_source(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
        self.log_dir = log_dir
        self.algo = getattr(args, 'algo', 'ferret')

        self.tokenizer.model_max_length = self.args.max_length
        special_tokens = dict()
        if self.tokenizer.pad_token is None:
            special_tokens['pad_token'] = DefaultToken.PAD_TOKEN.value
        if self.tokenizer.eos_token is None:
            special_tokens['eos_token'] = DefaultToken.EOS_TOKEN.value
        if self.tokenizer.bos_token is None:
            special_tokens['bos_token'] = DefaultToken.BOS_TOKEN.value
        if self.tokenizer.unk_token is None:
            special_tokens['unk_token'] = DefaultToken.UNK_TOKEN.value
        self.tokenizer.add_special_tokens(special_tokens)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map='cpu',
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.seed_pool = {seed: 0.0 for seed in self.candidate_seeds}
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.args.device}')
        else:
            self.device = torch.device('cpu')

        # FedSubMuon global states
        self.x_global = {}
        self.m_global = {}
        self.v_global = {}
        self.seeds = {}
        self.submuon_layer_dims = {}
        self.best_metric = math.inf if self.args.eval_metric == 'loss' else -math.inf
        self.seed_rng = np.random.RandomState(self.args.seed + 2026)

        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            self.x_global, self.m_global, self.seeds = init_submuon_state(self.model, self.args.rank_r, self.args.seed)
            if self.algo == 'fedsubsgd':
                self.m_global = {}
            if self.algo == 'fedsubadam':
                self.v_global = {k: torch.zeros_like(v) for k, v in self.x_global.items()}
            name_to_module = dict(self.model.named_modules())
            for layer_name in self.seeds.keys():
                module = name_to_module[layer_name]
                self.submuon_layer_dims[layer_name] = (module.out_features, module.in_features)

    def get_submuon_broadcast_state(self):
        return {
            'x_global': {k: v.clone() for k, v in self.x_global.items()},
            'm_global': {k: v.clone() for k, v in self.m_global.items()} if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
            'v_global': {k: v.clone() for k, v in self.v_global.items()} if self.algo == 'fedsubadam' else None,
            'seeds': dict(self.seeds),
        }

    def maybe_refresh_submuon_seeds(self, cur_round):
        if self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            return
        if getattr(self.args, 'stop_F', -1) > 0 and cur_round >= int(self.args.stop_F):
            return
        if self.args.seed_refresh_F <= 0:
            return
        if cur_round % self.args.seed_refresh_F != 0:
            return

        old_seeds = dict(self.seeds)
        new_seeds = {}
        for layer_name in self.seeds.keys():
            new_seeds[layer_name] = int(self.seed_rng.randint(1, 2**31 - 1))

        transport_state(
            x_global=self.x_global,
            m_global=self.m_global if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
            old_seeds=old_seeds,
            new_seeds=new_seeds,
            layer_dims=self.submuon_layer_dims,
            rank=self.args.rank_r,
            v_global=self.v_global if self.algo == 'fedsubadam' else None,
        )
        self.seeds = new_seeds

    def aggregate_submuon(self, client_payloads, selected_client_list):
        if self.args.equal_weight:
            weight_array = np.array([1.0 for _ in selected_client_list], dtype=np.float64)
            weight_array /= float(len(selected_client_list))
        else:
            weight_array = np.array([len(client.train_loader) for client in selected_client_list], dtype=np.float64)
            weight_array /= float(np.sum(weight_array))

        new_x = {name: torch.zeros_like(val) for name, val in self.x_global.items()}
        new_m = {name: torch.zeros_like(val) for name, val in self.m_global.items()} if self.algo in ['fedsubmuon', 'fedsubadam'] else None
        new_v = {name: torch.zeros_like(val) for name, val in self.v_global.items()} if self.algo == 'fedsubadam' else None

        for client_idx, payload in enumerate(client_payloads):
            w = float(weight_array[client_idx])
            for name in new_x.keys():
                new_x[name] += payload['x'][name].to(dtype=new_x[name].dtype) * w
                if self.algo in ['fedsubmuon', 'fedsubadam']:
                    new_m[name] += payload['m'][name].to(dtype=new_m[name].dtype) * w
                if self.algo == 'fedsubadam':
                    new_v[name] += payload['v'][name].to(dtype=new_v[name].dtype) * w

        self.x_global = new_x
        if self.algo in ['fedsubmuon', 'fedsubadam']:
            self.m_global = new_m
        if self.algo == 'fedsubadam':
            self.v_global = new_v

    def aggregate_seed_pool(self, selected_client_list, cur_round=1):
        for seed in self.candidate_seeds:
            self.seed_pool[seed] *= self.args.momentum

        if self.args.equal_weight:
            weight_array = np.array([1.0 for _ in selected_client_list], dtype=np.float64)
            weight_array /= float(len(selected_client_list))
        else:
            weight_array = np.array([len(client.train_loader) for client in selected_client_list], dtype=np.float64)
            weight_array /= float(np.sum(weight_array))
        for client_idx in range(len(selected_client_list)):
            local_seed_pool = selected_client_list[client_idx].local_seed_pool
            for seed, grad in local_seed_pool.items():
                self.seed_pool[seed] += grad * weight_array[client_idx]

        for client in selected_client_list:
            client.clear_model()

    def update_global_model_by_seed_pool(self):
        self.model.to(self.device)

        framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)

        progress_bar = tqdm(range(len(self.seed_pool)))

        # pull the latest model via accumulated {seed, grad} pairs on the server
        for seed, grad in self.seed_pool.items():
            framework.update(seed=seed, grad=grad)
            progress_bar.update(1)
            progress_bar.set_description('server update global model')

        self.model.to('cpu')

    def save_best_submuon_ckpt(self, metric, cur_round):
        if self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd'] or not self.args.save:
            return False

        improved = (metric < self.best_metric) if self.args.eval_metric == 'loss' else (metric > self.best_metric)
        if not improved:
            return False

        self.best_metric = metric
        os.makedirs(self.args.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.args.output_dir, 'best.pt')
        torch.save(
            {
                'backbone_state_dict': self.model.state_dict(),
                'x_global': {k: v.cpu() for k, v in self.x_global.items()},
                'm_global': {k: v.cpu() for k, v in self.m_global.items()} if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
                'v_global': {k: v.cpu() for k, v in self.v_global.items()} if self.algo == 'fedsubadam' else None,
                'seeds': dict(self.seeds),
                'round': int(cur_round),
                'best_metric': float(metric),
                'hparams': {
                    'algo': self.args.algo,
                    'rank_r': self.args.rank_r,
                    'beta': self.args.beta,
                    'beta1': getattr(self.args, 'beta1', None),
                    'beta2': getattr(self.args, 'beta2', None),
                    'eps': getattr(self.args, 'eps', None),
                    'lr': self.args.lr,
                    'ns_steps': self.args.ns_steps,
                    'seed_refresh_F': self.args.seed_refresh_F,
                },
            },
            ckpt_path,
        )
        return True

    def eval(self, cur_round, eval_avg_acc):
        if self.args.eval_metric == 'loss':
            eval_metric = self.eval_loss(cur_round)
        else:
            eval_metric = self.eval_generate(cur_round)

        if self.args.save and self.algo not in ['fedsubmuon', 'fedsubadam', 'fedsubsgd'] and cur_round > 0:
            save_dir = self.log_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            if (self.args.eval_metric == 'loss' and eval_metric < np.min(eval_avg_acc)) or (
                self.args.eval_metric != 'none' and eval_metric > np.max(eval_avg_acc)
            ):
                for file_name in os.listdir(save_dir):
                    if 'best' in file_name:
                        os.remove(os.path.join(save_dir, file_name))
                torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_best_round{cur_round}.bin'))
            for file_name in os.listdir(save_dir):
                if 'final' in file_name:
                    os.remove(os.path.join(save_dir, file_name))
            torch.save(self.model.state_dict(), os.path.join(save_dir, f'model_state_dict_final_round{cur_round}.bin'))
        return eval_metric

    def eval_loss(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()

        framework = None
        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                self.x_global,
                self.m_global if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
                self.seeds,
                trainable=False,
                v_state=self.v_global if self.algo == 'fedsubadam' else None,
            )

        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        loss_total_eval = 0.0
        num_eval = 0

        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }
                outputs = self.model(**batch)
                loss = outputs.loss
                progress_bar_eval.update(1)
                if torch.isnan(loss):
                    continue
                loss_total_eval += loss
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, loss: {loss_total_eval / num_eval}')
        print()
        print()

        if framework is not None:
            framework.clear_submuon_state()
        self.model = self.model.cpu()
        return (loss_total_eval / num_eval).item()

    def eval_generate(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()

        framework = None
        if self.algo in ['fedsubmuon', 'fedsubadam', 'fedsubsgd']:
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                self.x_global,
                self.m_global if self.algo in ['fedsubmuon', 'fedsubadam'] else None,
                self.seeds,
                trainable=False,
                v_state=self.v_global if self.algo == 'fedsubadam' else None,
            )

        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        acc_total_eval = 0.0
        num_eval = 0

        with torch.inference_mode():
            for batch in self.eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                label_ids = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_eval += rouge_score(output_ids[0][len(input_ids[0]):], label_ids[0], self.tokenizer)
                progress_bar_eval.update(1)
                num_eval += len(batch['input_ids'])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar_eval.set_description(f'eval at round {cur_round}, metric: {acc_total_eval / num_eval}')
        print()
        print()

        if framework is not None:
            framework.clear_submuon_state()
        self.model = self.model.cpu()
        return acc_total_eval / num_eval
