"""Learning-rate scheduler presets."""
import math

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ExponentialLR,
    LambdaLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)

from odyssey.training.callback import Callback

def warmup_cosine_lambda(warmup_steps, total_steps, min_lr_ratio=0.0):
    "LambdaLR multiplier: linear warmup then cosine decay to `min_lr_ratio`."
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)
    def f(step):
        if step < warmup_steps: return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return f

_SCHEDULER_PRESETS = frozenset({
    'onecycle', 'step', 'multistep', 'exponential', 'cosine', 'cosine_warmup', 'plateau', 'none',
})

def build_scheduler(scheduler, opt, *, n_epochs, steps_per_epoch, lr, **kwargs):
    "Build a PyTorch LR scheduler from a preset name; pass `None` or `'none'` to disable."
    if scheduler is None or scheduler == 'none': return None
    if not isinstance(scheduler, str):
        raise TypeError(f"scheduler must be a preset name, got {type(scheduler)}")
    if scheduler not in _SCHEDULER_PRESETS:
        raise ValueError(f"Unknown scheduler {scheduler!r}; choose from {sorted(_SCHEDULER_PRESETS)}")
    min_lr = kwargs.pop('min_lr', 1e-6)
    if scheduler == 'onecycle':
        max_lr = kwargs.pop('max_lr', lr)
        total_steps = n_epochs * steps_per_epoch
        return OneCycleLR(
            opt, max_lr=max_lr, total_steps=total_steps,
            pct_start=kwargs.pop('pct_start', 0.3),
            anneal_strategy=kwargs.pop('anneal_strategy', 'cos'),
            div_factor=kwargs.pop('div_factor', max(1.0, max_lr / lr)),
            final_div_factor=kwargs.pop('final_div_factor', max(1.0, max_lr / max(min_lr, 1e-12))),
            **kwargs,
        )
    if scheduler == 'step':
        return StepLR(opt, step_size=kwargs.pop('step_size', max(1, n_epochs // 3)),
                      gamma=kwargs.pop('gamma', 0.1), **kwargs)
    if scheduler == 'multistep':
        m1 = max(1, int(0.5 * n_epochs))
        m2 = max(m1 + 1, int(0.75 * n_epochs))
        milestones = kwargs.pop('milestones', [m1, m2])
        return MultiStepLR(opt, milestones=milestones, gamma=kwargs.pop('gamma', 0.1), **kwargs)
    if scheduler == 'exponential':
        return ExponentialLR(opt, gamma=kwargs.pop('gamma', 0.95), **kwargs)
    if scheduler == 'cosine':
        return CosineAnnealingLR(opt, T_max=n_epochs, eta_min=min_lr, **kwargs)
    if scheduler == 'cosine_warmup':
        total_steps = n_epochs * steps_per_epoch
        warmup_ratio = kwargs.pop('warmup_ratio', 0.1)
        warmup_steps = int(warmup_ratio * total_steps)
        lr_lambda = warmup_cosine_lambda(
            warmup_steps, total_steps,
            min_lr_ratio=(min_lr / lr) if lr > 0 else 0.0,
        )
        return LambdaLR(opt, lr_lambda=lr_lambda, **kwargs)
    if scheduler == 'plateau':
        return ReduceLROnPlateau(
            opt, mode='min', factor=kwargs.pop('factor', 0.5),
            patience=kwargs.pop('patience', 1), threshold=kwargs.pop('threshold', 1e-3),
            min_lr=min_lr, **kwargs,
        )
    return None



class SchedulerCB(Callback):
    "Step LR scheduler after each optimizer step (OneCycle / warmup cosine) or each valid epoch (others)."
    order = 1

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._per_batch = isinstance(scheduler, (OneCycleLR, LambdaLR))

    def before_fit(self, learn):
        learn.sched = self.scheduler

    def after_optimizer_step(self, learn):
        if self._per_batch and learn.training:
            self.scheduler.step()

    def after_epoch(self, learn):
        if learn.training:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            metric = None
            if hasattr(learn, "metrics"):
                metric = learn.metrics.all_metrics["loss"].compute()
            if metric is not None:
                self.scheduler.step(metric)
        elif not self._per_batch:
            self.scheduler.step()
