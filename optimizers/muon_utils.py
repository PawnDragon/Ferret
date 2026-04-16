import importlib
import inspect

import torch

from optimizers.submuon_utils import zeropower_via_newtonschulz5


_MUON_LOGGED_MESSAGES = set()


def _log_muon_once(message):
    if message in _MUON_LOGGED_MESSAGES:
        return
    print(message)
    _MUON_LOGGED_MESSAGES.add(message)


class LocalMuon(torch.optim.Optimizer):
    """
    Minimal Muon-compatible optimizer for 2D trainable matrices.

    LoRA A/B weights are matrices, so this covers FedIT's intended Muon path.
    Parameters with fewer than 2 dims fall back to momentum SGD to keep the
    optimizer robust if a future PEFT module adds a bias-like trainable tensor.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        weight_decay=0.0,
        ns_steps=5,
        nesterov=True,
    ):
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        defaults = dict(
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            ns_steps=int(ns_steps),
            nesterov=bool(nesterov),
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group['lr'])
            momentum = float(group['momentum'])
            weight_decay = float(group['weight_decay'])
            ns_steps = int(group['ns_steps'])
            nesterov = bool(group['nesterov'])

            for param in group['params']:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')

                if weight_decay != 0.0:
                    param.data.mul_(1.0 - lr * weight_decay)

                grad_f32 = grad.to(dtype=torch.float32)
                state = self.state[param]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad_f32)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(grad_f32)
                update = grad_f32.add(buf, alpha=momentum) if nesterov else buf

                if update.ndim >= 2:
                    matrix_update = update.reshape(update.shape[0], -1)
                    matrix_update = zeropower_via_newtonschulz5(matrix_update, ns_steps)
                    scale = max(1.0, float(matrix_update.shape[0]) / float(matrix_update.shape[1])) ** 0.5
                    update = matrix_update.reshape_as(update).mul_(scale)

                param.data.add_(update.to(device=param.device, dtype=param.dtype), alpha=-lr)

        return loss


def _resolve_external_muon_module():
    try:
        return importlib.import_module('muon')
    except ImportError:
        return None


def _filter_supported_kwargs(optimizer_cls, kwargs):
    try:
        signature = inspect.signature(optimizer_cls)
    except (TypeError, ValueError):
        return kwargs

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    aliases = {
        'ns_steps': ['ns_steps', 'newton_schulz_steps', 'ns_iters'],
        'nesterov': ['nesterov', 'use_nesterov'],
    }

    filtered = {}
    for key, value in kwargs.items():
        if key in parameters:
            filtered[key] = value
            continue
        for alias in aliases.get(key, []):
            if alias in parameters:
                filtered[alias] = value
                break
    return filtered


def _instantiate_optimizer_class(optimizer_cls, params, kwargs):
    filtered_kwargs = _filter_supported_kwargs(optimizer_cls, kwargs)
    attempts = [
        filtered_kwargs,
        {k: v for k, v in filtered_kwargs.items() if k in ['lr', 'momentum', 'weight_decay']},
        {k: v for k, v in filtered_kwargs.items() if k in ['lr', 'momentum']},
        {k: v for k, v in filtered_kwargs.items() if k == 'lr'},
        {},
    ]

    last_error = None
    for attempt_kwargs in attempts:
        try:
            return optimizer_cls(params, **attempt_kwargs)
        except TypeError as exc:
            last_error = exc
            continue
    raise last_error


def _split_muon_params(params):
    muon_params = []
    aux_params = []
    for param in params:
        if getattr(param, 'ndim', 0) >= 2:
            muon_params.append(param)
        else:
            aux_params.append(param)
    return muon_params, aux_params


def _build_muon_with_aux_adam(muon_with_aux_adam_cls, muon_params, aux_params, kwargs):
    param_groups = []
    if len(muon_params) > 0:
        param_groups.append(
            dict(
                params=muon_params,
                use_muon=True,
                lr=float(kwargs['lr']),
                weight_decay=float(kwargs['weight_decay']),
            )
        )
    if len(aux_params) > 0:
        param_groups.append(
            dict(
                params=aux_params,
                use_muon=False,
                lr=float(kwargs['lr']),
                betas=(0.9, 0.95),
                weight_decay=float(kwargs['weight_decay']),
            )
        )
    return muon_with_aux_adam_cls(param_groups)


def _build_external_muon_optimizer(params, kwargs):
    muon_module = _resolve_external_muon_module()
    if muon_module is None:
        return None, None

    muon_params, aux_params = _split_muon_params(params)
    single_device_muon_cls = getattr(muon_module, 'SingleDeviceMuon', None)
    if single_device_muon_cls is not None and len(muon_params) == len(params):
        optimizer = _instantiate_optimizer_class(single_device_muon_cls, muon_params, kwargs)
        return optimizer, 'muon.SingleDeviceMuon'

    muon_with_aux_adam_cls = getattr(muon_module, 'MuonWithAuxAdam', None)
    if muon_with_aux_adam_cls is not None:
        optimizer = _build_muon_with_aux_adam(muon_with_aux_adam_cls, muon_params, aux_params, kwargs)
        return optimizer, 'muon.MuonWithAuxAdam'

    if single_device_muon_cls is not None and len(aux_params) == 0:
        optimizer = _instantiate_optimizer_class(single_device_muon_cls, muon_params, kwargs)
        return optimizer, 'muon.SingleDeviceMuon'

    return None, None


def build_muon_optimizer(params, lr, momentum, weight_decay, ns_steps, nesterov=True):
    params = list(params)
    if len(params) == 0:
        return None

    kwargs = {
        'lr': float(lr),
        'momentum': float(momentum),
        'weight_decay': float(weight_decay),
        'ns_steps': int(ns_steps),
        'nesterov': bool(nesterov),
    }

    if _resolve_external_muon_module() is not None:
        try:
            optimizer, source = _build_external_muon_optimizer(params, kwargs)
            if optimizer is not None:
                _log_muon_once(f'[muon] using external Muon optimizer: {source}')
                return optimizer
            _log_muon_once('[warn][muon] installed muon package has no SingleDeviceMuon or MuonWithAuxAdam')
        except Exception as exc:
            _log_muon_once(f'[warn][muon] failed to build external Muon optimizer: {exc}')

    _log_muon_once('[muon] using local Muon optimizer implementation')
    return LocalMuon(params, **kwargs)
