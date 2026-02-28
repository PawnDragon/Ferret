import math


class AdaptiveRebaseController(object):
    def __init__(
        self,
        best_loss,
        best_round=0,
        min_delta=1e-4,
        rebase_patience=5,
        rebase_cooldown=3,
        max_rebase=2,
        enable_rebase=True,
        enable_early_stop=False,
        early_stop_patience=15,
    ):
        self.best_loss = float(best_loss) if math.isfinite(float(best_loss)) else float('inf')
        self.best_round = int(best_round)
        self.min_delta = float(min_delta)
        self.rebase_patience = max(int(rebase_patience), 1)
        self.rebase_cooldown = max(int(rebase_cooldown), 0)
        self.max_rebase = max(int(max_rebase), 0)
        self.enable_rebase = bool(enable_rebase)
        self.enable_early_stop = bool(enable_early_stop)
        self.early_stop_patience = max(int(early_stop_patience), 1)

        self.no_improve = 0
        self.rebase_count = 0
        self.cooldown_left = 0

    def _is_improved(self, curr_loss):
        return (self.best_loss - float(curr_loss)) > self.min_delta

    def snapshot(self):
        return {
            'best_loss': float(self.best_loss),
            'best_round': int(self.best_round),
            'no_improve': int(self.no_improve),
            'rebase_count': int(self.rebase_count),
            'cooldown_left': int(self.cooldown_left),
        }

    def step(self, cur_round, curr_loss):
        result = {
            'improved': False,
            'did_rebase': False,
            'should_early_stop': False,
            'phase': 'noop',
        }
        if not math.isfinite(float(curr_loss)):
            result['phase'] = 'invalid_eval'
            result.update(self.snapshot())
            return result

        cur_round = int(cur_round)
        curr_loss = float(curr_loss)
        improved = self._is_improved(curr_loss)

        # Step A: cooldown has highest priority for control actions.
        if self.cooldown_left > 0:
            if improved:
                self.best_loss = curr_loss
                self.best_round = cur_round
                result['improved'] = True
            self.cooldown_left -= 1
            result['phase'] = 'cooldown'
            result.update(self.snapshot())
            return result

        # Step B: regular progress accounting.
        if improved:
            self.best_loss = curr_loss
            self.best_round = cur_round
            self.no_improve = 0
            result['improved'] = True
        else:
            self.no_improve += 1

        if self.enable_rebase and self.no_improve >= self.rebase_patience and self.rebase_count < self.max_rebase:
            self.rebase_count += 1
            self.cooldown_left = self.rebase_cooldown
            self.no_improve = 0  # restart-style reset
            result['did_rebase'] = True
            result['phase'] = 'rebase'
            result.update(self.snapshot())
            return result

        if self.enable_early_stop:
            rebase_exhausted = (not self.enable_rebase) or (self.rebase_count >= self.max_rebase)
            if rebase_exhausted and self.no_improve >= self.early_stop_patience:
                result['should_early_stop'] = True
                result['phase'] = 'early_stop'
            else:
                result['phase'] = 'normal'
        else:
            result['phase'] = 'normal'

        result.update(self.snapshot())
        return result
