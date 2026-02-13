from tqdm import tqdm
from copy import deepcopy

from optimizers.ferret_optimizer import *


class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None

        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{args.device}')
        else:
            self.device = torch.device('cpu')
        self.candidate_seeds = candidate_seeds

    def _next_batch(self):
        try:
            batch = next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.train_loader)
            batch = next(self.train_iterator)
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'labels': batch['labels'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
        }

    def local_train_with_seed_pool(self, pulled_model, cur_round, submuon_state=None):
        self.model = pulled_model
        self.model.to(self.device)

        if getattr(self.args, 'algo', 'ferret') == 'fedsubmuon':
            framework = FerretFramework(self.model, args=self.args, lr=self.args.lr, candidate_seeds=self.candidate_seeds)
            framework.set_submuon_state(
                x_state=submuon_state['x_global'],
                m_state=submuon_state['m_global'],
                seeds=submuon_state['seeds'],
                trainable=True,
            )
            self.model.train()

            loss_total_train = 0.0
            num_trained = 0

            if self.args.batch_or_epoch == 'epoch':
                iter_steps = len(self.train_loader)
                progress_bar = tqdm(range(iter_steps))
                for cur_step, batch in enumerate(self.train_loader):
                    batch = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'labels': batch['labels'].to(self.device),
                        'attention_mask': batch['attention_mask'].to(self.device),
                    }
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )
            else:
                iter_steps = max(int(self.args.local_step), 1)
                progress_bar = tqdm(range(iter_steps))
                for cur_step in range(iter_steps):
                    batch = self._next_batch()
                    apply_optim_step = (cur_step % self.args.n_accum == self.args.n_accum - 1) or (cur_step == iter_steps - 1)
                    _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                    progress_bar.update(1)
                    if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                        loss_total_train += loss
                        num_trained += len(batch['input_ids'])
                    progress_bar.set_description(
                        f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                    )

            x_local, m_local = framework.export_submuon_state()
            framework.clear_submuon_state()
            self.model = None
            return {
                'x': x_local,
                'm': m_local,
                'loss': float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0,
            }

        old_params = [(name, deepcopy(param.data)) for name, param in self.model.named_parameters() if param.requires_grad]

        # initialize a seed pool
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        lr = self.args.lr

        if self.args.batch_or_epoch == 'epoch':
            iter_steps = len(self.train_loader)
        else:
            iter_steps = self.args.local_step

        # Ferret Framework
        framework = FerretFramework(self.model, args=self.args, lr=lr, candidate_seeds=self.candidate_seeds)
        self.model.train()
        self.model.zero_grad()
        loss_total_train = 0.0
        num_trained = 0

        if self.args.batch_or_epoch == 'epoch':
            progress_bar = tqdm(range(iter_steps))
            for cur_step, batch in enumerate(self.train_loader):
                batch = {
                    'input_ids': batch['input_ids'].to(self.device),
                    'labels': batch['labels'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                }
                if cur_step % self.args.n_accum == self.args.n_accum - 1:
                    apply_optim_step = True
                else:
                    apply_optim_step = False
                _, loss = framework.step(batch, apply_optim_step=apply_optim_step)
                progress_bar.update(1)
                if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                    loss_total_train += loss
                    num_trained += len(batch['input_ids'])
                progress_bar.set_description(
                    f'client {self.idx} train at epoch {cur_round}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                )
        else:
            progress_bar = tqdm(range(iter_steps))
            for cur_step in range(iter_steps):
                batch = self._next_batch()

                if cur_step % self.args.n_accum == self.args.n_accum - 1:
                    apply_optim_step = True
                else:
                    apply_optim_step = False

                _, loss = framework.step(batch, apply_optim_step=apply_optim_step)

                progress_bar.update(1)
                if (not torch.isnan(loss)) and (self.args.grad_clip <= 0 or loss != 0.0):
                    loss_total_train += loss
                    num_trained += len(batch['input_ids'])
                progress_bar.set_description(
                    f'client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}'
                )

        self.local_seed_pool = framework.project_update(dict(old_params))

        # save both CPU and GPU memory
        del old_params, framework
        self.model = None
        return float((loss_total_train / num_trained).item()) if num_trained != 0 else 0.0

    def clear_model(self):
        # clear model to save memory
        self.model = None

    def migrate(self, device):
        """
        migrate a client to a new device
        """
        self.device = device

    def pull(self, forked_global_model):
        """
        pull model from the server
        """
        self.model = forked_global_model
