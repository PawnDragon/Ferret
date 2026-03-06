import torch


class SubspacesAllocator(object):
    """
    Local copy adapted from AdaMSS (adamss_pkg/asa.py).
    Only the score-tracking methods used by FedMultiSubMuon are kept here.
    """

    def __init__(
        self,
        tt,
        target_KK,
        init_warmup: int,
        final_warmup: int,
        mask_interval: int,
        beta1: float,
        beta2: float,
        total_step=None,
    ):
        super(SubspacesAllocator, self).__init__()
        self.target_KK = target_KK
        self.initial_warmup = init_warmup
        self.final_warmup = final_warmup
        self.mask_interval = mask_interval
        self.beta1 = beta1
        self.beta2 = beta2
        self.total_step = total_step
        self.total_KK = None
        self.curr_KK = None
        self.tt = tt
        self.ipt = {}
        self.exp_avg_ipt = {}
        self.exp_avg_unc = {}
        self.cat_ipt = {}

        assert self.beta1 < 1 and self.beta1 > 0
        assert self.beta2 < 1 and self.beta2 > 0

    def update_ipt(self, model):
        for name, param in model.named_parameters():
            if "adamss_" in name and param.grad is not None:
                if name not in self.ipt:
                    self.ipt[name] = torch.zeros_like(param)
                    self.exp_avg_ipt[name] = torch.zeros_like(param)
                    self.exp_avg_unc[name] = torch.zeros_like(param)
                with torch.no_grad():
                    self.ipt[name] = (param * param.grad).abs().detach()
                    self.exp_avg_ipt[name] = self.beta1 * self.exp_avg_ipt[name] + (1 - self.beta1) * self.ipt[name]
                    self.exp_avg_unc[name] = self.beta2 * self.exp_avg_unc[name] + (
                        1 - self.beta2
                    ) * (self.ipt[name] - self.exp_avg_ipt[name]).abs()

    def calculate_score(self, name, param=None, metric="ipt"):
        if metric == "ipt":
            return self.exp_avg_ipt[name] * self.exp_avg_unc[name]
        if metric == "mag":
            if param is None:
                raise ValueError("param is required when metric='mag'")
            return param.abs().detach().clone()
        raise ValueError("Unexcptected Metric: %s" % metric)
