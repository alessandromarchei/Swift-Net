import math

import torch
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLR(LRScheduler):
    """Linear warmup followed by cosine annealing, stepped once per optimizer step.

    The schedule needs the total number of optimizer steps, which depends on the dataloader
    length, the number of devices and ``accumulate_grad_batches``. These are only known once
    the Lightning trainer exists, so the LightningModule calls :meth:`set_total_steps` from
    ``configure_optimizers``. Until then the learning rate is the warmup start value.

    Args:
        optimizer: wrapped optimizer.
        warmup_epochs (int): epochs of linear warmup from ``warmup_start_factor * base_lr`` to ``base_lr``.
        min_lr (float): learning rate reached at the last step of the cosine phase.
        warmup_start_factor (float): fraction of ``base_lr`` used at the first step.
    """

    def __init__(self, optimizer, warmup_epochs=5, min_lr=1e-6, warmup_start_factor=0.01, last_epoch=-1):
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be a non-negative integer, got {warmup_epochs!r}")
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.warmup_start_factor = warmup_start_factor
        self.total_steps = None
        self.warmup_steps = None
        super().__init__(optimizer, last_epoch)

    def set_total_steps(self, total_steps, max_epochs):
        """Fix the step budget: ``total_steps`` optimizer steps spread over ``max_epochs`` epochs."""
        total_steps = int(total_steps)
        steps_per_epoch = total_steps // max_epochs
        warmup_steps = self.warmup_epochs * steps_per_epoch
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_epochs ({self.warmup_epochs}) must be smaller than training.epochs ({max_epochs})"
            )
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        # Apply the lr of the current step now that the schedule is defined
        for group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _factor(self, step, base_lr):
        if self.total_steps is None or step < self.warmup_steps:
            warmup = self.warmup_steps or 1
            start = self.warmup_start_factor
            return start + (1.0 - start) * min(step, warmup) / warmup
        progress = min(step - self.warmup_steps, self.cosine_steps) / self.cosine_steps
        min_factor = self.min_lr / base_lr
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    @property
    def cosine_steps(self):
        return max(1, self.total_steps - self.warmup_steps)

    def get_lr(self):
        step = max(self.last_epoch, 0)
        return [base_lr * self._factor(step, base_lr) for base_lr in self.base_lrs]


# Schedulers defined here, looked up by name before falling back to torch.optim.lr_scheduler
_SCHEDULERS = {"WarmupCosineLR": WarmupCosineLR}


def make_scheduler(sche_name, optimizer, **sche_config):
    """Build a scheduler from its name: a scheduler of this module or any ``torch.optim.lr_scheduler`` class."""
    if sche_name in _SCHEDULERS:
        return _SCHEDULERS[sche_name](optimizer, **sche_config)
    return getattr(torch.optim.lr_scheduler, sche_name)(optimizer=optimizer, **sche_config)
