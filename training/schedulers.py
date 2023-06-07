from torch.optim.optimizer import Optimizer
from utils.conf_utils import get_obj_from_str
from torch.optim.lr_scheduler import _LRScheduler
import torch

class KneeLRScheduler(_LRScheduler):
    def __init__(self, optimizer, peak_lr, warmup_steps=0, explore_steps=0, total_steps=0, min_lr=0):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.explore_steps = explore_steps
        self.total_steps = total_steps
        self.decay_steps = self.total_steps - (self.explore_steps + self.warmup_steps)
        self.current_step = 1
        self.min_lr = min_lr

        assert self.decay_steps >= 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

    def get_lr(self, global_step):
        if global_step <= self.warmup_steps:
            return self.peak_lr * global_step / self.warmup_steps
        elif global_step <= (self.explore_steps + self.warmup_steps):
            return self.peak_lr
        else:
            slope = -1 * self.peak_lr / self.decay_steps
            return max(self.min_lr, self.peak_lr + slope * (global_step - (self.explore_steps + self.warmup_steps)))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

class StepLRScheduler:
    def __init__(self, optimizer, peak_lr, gamma=0.5, step_size=200, max_epochs=1000):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.gamma = gamma
        self.max_epochs = max_epochs
        self.step_size = step_size
        self.current_step = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

        if not isinstance(self.optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(self.optimizer).__name__))

    def get_lr(self, global_step):
        factor = global_step // self.step_size
        lr = self.peak_lr * (self.gamma ** factor)
        return lr

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

class MultiStepLR_Restart(_LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            restarts=None,
            weights=None,
            gamma=0.1,
            clear_state=False,
            last_epoch=-1,
    ):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.clear_state = clear_state
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(MultiStepLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            if self.clear_state:
                self.optimizer.state = defaultdict(dict)
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            # print(self.optimizer.param_groups)
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def step(self):
        self.current_step += 1
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_lr(self.current_step)

class CosineAnnealingLR_Restart(_LRScheduler):
    def __init__(
            self, optimizer, T_period, restarts=None, weights=None, eta_min=0, last_epoch=-1
    ):
        self.T_period = T_period
        self.T_max = self.T_period[0]  # current T period
        self.eta_min = eta_min
        self.restarts = restarts if restarts else [0]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        self.current_step = 1
        assert len(self.restarts) == len(
            self.restart_weights
        ), "restarts and their weights do not match."
        super(CosineAnnealingLR_Restart, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [
                group["initial_lr"] * weight for group in self.optimizer.param_groups
            ]
        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (
                2 * self.T_max
        ) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max))
            / (
                    1
                    + math.cos(
                math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max
            )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

if __name__ == '__main__':
    optimizer = torch.optim.Adam(
        [torch.zeros(3, 64, 3, 3)], lr=2e-4, weight_decay=0, betas=(0.9, 0.99)
    )
    scheduler = CosineAnnealingLR_Restart(
        optimizer, T_period=[155, 155, 155, 155, 155, 155], eta_min=1e-7, restarts=[155, 310, 465, 620, 775],
        weights=[1, 1, 1, 1, 1]
    )