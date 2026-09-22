"""Learner training loop."""
import math
import warnings
from contextlib import contextmanager
from functools import partial

import fastcore.all as fc
from fastcore.basics import patch

from odyssey.training.callback import CancelBatchException, CancelEpochException, CancelFitException, run_callbacks
from odyssey.training.callbacks import GradAccumCB, LRFinder, LRFind, TrainCB
from odyssey.tracking.diagnostics import describe_model
from odyssey.training.schedule import SchedulerCB, build_scheduler
from odyssey.training.weights import _merge_init_cbs


def _steps_per_epoch(cbs, dls) -> int:
    "Optimizer steps per train epoch; accounts for GradAccumCB when present."
    n_batches = len(dls.train)
    for cb in cbs:
        if isinstance(cb, GradAccumCB) and cb.n_accum > 1:
            n_accum = cb.n_accum
            steps = n_batches // n_accum
            if cb.step_on_epoch_end and n_batches % n_accum:
                steps += 1
            return steps
    return n_batches


def _has_train_cb(cbs) -> bool:
    return any(hasattr(cb, "predict") for cb in cbs)


def _ensure_train_cb(cbs, *, auto_train: bool):
    cbs = list(cbs)
    if _has_train_cb(cbs):
        return cbs
    if not auto_train:
        raise ValueError("Learner requires a train callback with `predict`; pass TrainCB or set auto_train=True")
    return cbs + [TrainCB()]


class Learner:
    def __init__(
        self,
        model,
        dls,
        loss_func,
        opt_func,
        lr=0.01,
        cbs=None,
        init=None,
        lr_find=None,
        scheduler="onecycle",
        scheduler_kwargs=None,
        pin_memory=None,
        project_name=None,
        auto_train=True,
    ):
        cbs = _merge_init_cbs(cbs, init)
        cbs = _ensure_train_cb(cbs, auto_train=auto_train)
        skw = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
        fc.store_attr()
        self.scheduler_kwargs = skw

    def callbacks(self, method):
        return run_callbacks(self.cbs, method, self)

    @contextmanager
    def cb_ctx(self, name, cancel_exc):
        self.callbacks(f"before_{name}")
        try:
            yield
        except cancel_exc:
            pass

    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        with self.cb_ctx("epoch", CancelEpochException):
            for self.num, self.batch in enumerate(self.dl):
                with self.cb_ctx("batch", CancelBatchException):
                    self.one_batch()
                    self.callbacks("after_batch")
        self.callbacks("after_epoch")

    def one_batch(self):
        self.preds = self.predict()
        self.loss = self.get_loss()
        if self.model.training:
            self.callbacks("before_backward")
            self.backward()
            self.callbacks("after_backward")
            self._optimizer_step()

    def _optimizer_step(self):
        if not getattr(self, "should_step", True):
            return
        if not getattr(self, "_use_amp", False):
            self.clip_grad()
        self.step()
        self.zero_grad()
        self.callbacks("after_optimizer_step")

    def fit(self, n_epochs):
        self.opt = self.opt_func(self.model.parameters(), lr=self.lr)
        self.epochs = range(n_epochs)
        self.steps_per_epoch = _steps_per_epoch(self.cbs, self.dls)
        sched_cbs = []
        if self.scheduler and not any(isinstance(cb, LRFinder) for cb in self.cbs):
            sched = build_scheduler(
                self.scheduler,
                self.opt,
                n_epochs=n_epochs,
                steps_per_epoch=self.steps_per_epoch,
                lr=self.lr,
                **self.scheduler_kwargs,
            )
            if sched is not None:
                sched_cbs = [SchedulerCB(sched)]
        orig_cbs = self.cbs
        self.cbs = orig_cbs + sched_cbs
        with self.cb_ctx("fit", CancelFitException):
            try:
                for self.epoch in self.epochs:
                    self.one_epoch(True)
                    self.one_epoch(False)
            finally:
                self.callbacks("after_fit")
                self.cbs = orig_cbs

    def validate(self):
        "Run one validation epoch without training."
        self.epoch = getattr(self, "epoch", 0)
        self.one_epoch(False)

    def __getattr__(self, name):
        if name in ("predict", "get_loss", "backward", "clip_grad", "step", "zero_grad"):
            return partial(self.callbacks, name)
        raise AttributeError(f"Attribute {name} not found")

    @property
    def training(self):
        return self.model.training

    def find_lr(self, **kwargs):
        "Run LR sweep; uses attached `lr_find` when present, else a one-shot `LRFind`."
        if kwargs or self.lr_find is None:
            return LRFind(**kwargs).run(self)
        return self.lr_find.run(self)


class MomentumLearner(Learner):
    def __init__(
        self,
        model,
        dls,
        loss_func,
        opt_func,
        lr=0.01,
        cbs=None,
        init=None,
        lr_find=None,
        scheduler="onecycle",
        scheduler_kwargs=None,
        pin_memory=None,
        project_name=None,
        auto_train=False,
    ):
        warnings.warn(
            "MomentumLearner is deprecated; use Learner with TrainCB (auto-injected by default).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            model,
            dls,
            loss_func,
            opt_func,
            lr,
            cbs,
            init=init,
            lr_find=lr_find,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            pin_memory=pin_memory,
            project_name=project_name,
            auto_train=auto_train,
        )

    def predict(self):
        return self.model(self.batch[0])

    def get_loss(self):
        return self.loss_func(self.preds, self.batch[1])

    def backward(self):
        self.loss.backward()

    def step(self):
        self.opt.step()

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)


@patch
def describe(self: Learner, **kwargs):
    "Diagnostics for `self.model`; uses one train batch when no input is passed."
    if kwargs.get("ex_inp") is None and kwargs.get("inp_size") is None:
        b = self.dls.train.one_batch()
        kwargs["ex_inp"] = b[0][:1]
    return describe_model(self.model, **kwargs)
