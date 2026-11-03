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


def _as_step_count(name, value):
    """Read a step count from the YAML: ints, integral floats (``1.0e+5``) and numeric strings (``1e5``)."""
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer number of optimizer steps, got {value!r}") from None
    if not as_float.is_integer():
        raise ValueError(f"{name} must be an integer number of optimizer steps, got {value!r}")
    return int(as_float)


class ConstantCosineLR(LRScheduler):
    """Constant learning rate, then cosine annealing to ``min_lr``, stepped once per optimizer step.

    ``t`` is the global step, i.e. the number of optimizer steps already taken (``trainer.global_step``).
    The optimizer step with global step ``t`` uses

        t <  constant_steps:                 base_lr
        constant_steps <= t < total_steps:   min_lr + (base_lr - min_lr) / 2 * (1 + cos(pi * (t - constant_steps)
                                                                              / (total_steps - constant_steps)))
        t >= total_steps:                    min_lr

    ``base_lr`` is the optimizer lr of each param group (``optimizer.lr``, or ``videonet.visual_lr`` for the
    encoder group). The lr at ``t == constant_steps`` is still ``base_lr`` and the lr at ``t == total_steps``
    is exactly ``min_lr``.

    The scheduler's ``last_epoch`` equals ``t``: construction applies the lr of step 0, and Lightning calls
    ``step()`` after every real ``optimizer.step()`` (never during gradient accumulation), which moves to
    the lr of the next global step. The lr is a closed-form function of ``last_epoch``, so restoring the
    scheduler state from a checkpoint continues on the same curve.

    Args:
        optimizer: wrapped optimizer.
        constant_steps (int): optimizer steps at the initial lr.
        total_steps (int): global step at which ``min_lr`` is reached.
        min_lr (float): final learning rate, kept after ``total_steps``.
    """

    def __init__(self, optimizer, constant_steps, total_steps, min_lr=0.0, last_epoch=-1):
        constant_steps = _as_step_count("constant_steps", constant_steps)
        total_steps = _as_step_count("total_steps", total_steps)
        min_lr = float(min_lr)
        if not 0 <= constant_steps < total_steps:
            raise ValueError(
                f"Need 0 <= constant_steps < total_steps, got constant_steps={constant_steps}, "
                f"total_steps={total_steps}"
            )
        for i, group in enumerate(optimizer.param_groups):
            initial_lr = group.get("initial_lr", group["lr"])
            if not 0.0 <= min_lr <= initial_lr:
                raise ValueError(
                    f"Need 0 <= min_lr <= initial lr, got min_lr={min_lr} and lr={initial_lr} (param group {i})"
                )
        self.constant_steps = constant_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def lr_at(self, step, base_lr):
        """Learning rate used by the optimizer step with global step ``step``."""
        if step < self.constant_steps:
            return base_lr
        if step >= self.total_steps:
            return self.min_lr
        progress = (step - self.constant_steps) / (self.total_steps - self.constant_steps)
        return self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_lr(self):
        step = max(self.last_epoch, 0)
        return [self.lr_at(step, base_lr) for base_lr in self.base_lrs]

    def load_state_dict(self, state_dict):
        """Restore position and schedule from a checkpoint; report if the configured schedule differs.

        The checkpoint wins so that a resumed run continues exactly on the curve it was trained with.
        """
        configured = (self.constant_steps, self.total_steps, self.min_lr)
        super().load_state_dict(state_dict)
        restored = (self.constant_steps, self.total_steps, self.min_lr)
        if configured != restored:
            print(
                "ConstantCosineLR: the checkpoint schedule (constant_steps, total_steps, min_lr)="
                f"{restored} differs from the configured {configured}; continuing with the checkpoint schedule."
            )


# Schedulers defined here, looked up by name before falling back to torch.optim.lr_scheduler
_SCHEDULERS = {
    "WarmupCosineLR": WarmupCosineLR,
    "ConstantCosineLR": ConstantCosineLR,
    "constant_cosine": ConstantCosineLR,
}


def make_scheduler(sche_name, optimizer, **sche_config):
    """Build a scheduler from its name: a scheduler of this module or any ``torch.optim.lr_scheduler`` class."""
    if sche_name in _SCHEDULERS:
        return _SCHEDULERS[sche_name](optimizer, **sche_config)
    return getattr(torch.optim.lr_scheduler, sche_name)(optimizer=optimizer, **sche_config)
