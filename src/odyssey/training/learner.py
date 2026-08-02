"""Learner training loop."""
from contextlib import contextmanager
from functools import partial

import fastcore.all as fc
from fastcore.basics import patch

from odyssey.training.callback import CancelBatchException, CancelEpochException, CancelFitException, run_callbacks
from odyssey.training.callbacks import LRFinder, LRFind, TrainCB
from odyssey.tracking.diagnostics import describe_model
from odyssey.training.schedule import SchedulerCB, build_scheduler
from odyssey.training.weights import _merge_init_cbs

class Learner:
    def __init__(self, model, dls, loss_func, opt_func, lr=0.01, cbs=None, init=None, lr_find=None,
                 scheduler='onecycle', scheduler_kwargs=None):
        cbs = _merge_init_cbs(cbs, init)
        skw = {} if scheduler_kwargs is None else dict(scheduler_kwargs)
        fc.store_attr()
        self.scheduler_kwargs = skw
        
    def callbacks(self, method): run_callbacks(self.cbs, method, self)
    
    @contextmanager
    def cb_ctx(self, name, cancel_exc):
        self.callbacks(f'before_{name}')
        try: yield
        except cancel_exc: pass
        
    def one_epoch(self, train):
        self.model.training = train
        self.dl = self.dls.train if train else self.dls.valid
        with self.cb_ctx('epoch', CancelEpochException):
            for self.num, self.batch in enumerate(self.dl):
                with self.cb_ctx('batch', CancelBatchException):
                    self.one_batch()
                    self.callbacks('after_batch')
        self.callbacks('after_epoch')
        
    def one_batch(self):
        self.preds = self.predict()
        self.loss = self.get_loss()
        if self.model.training:
            self.backward()
            self.clip_grad()
            self.step()
            self.zero_grad()
        
    def fit(self, n_epochs):
        self.opt = self.opt_func(self.model.parameters(), lr=self.lr)
        self.epochs = range(n_epochs)
        sched_cbs = []
        if self.scheduler and not any(isinstance(cb, LRFinder) for cb in self.cbs):
            sched = build_scheduler(
                self.scheduler, self.opt,
                n_epochs=n_epochs, steps_per_epoch=len(self.dls.train), lr=self.lr,
                **self.scheduler_kwargs,
            )
            if sched is not None:
                sched_cbs = [SchedulerCB(sched)]
        orig_cbs = self.cbs
        self.cbs = orig_cbs + sched_cbs
        with self.cb_ctx('fit', CancelFitException):
            try:
                for self.epoch in self.epochs:
                    self.one_epoch(True)
                    self.one_epoch(False)
            finally:
                self.callbacks('after_fit')
                self.cbs = orig_cbs
        
    def __getattr__(self, name):
        if name in ('predict', 'get_loss', 'backward', 'clip_grad', 'step', 'zero_grad'):
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
    def __init__(self, model, dls, loss_func, opt_func, lr=0.01, cbs=None, init=None, lr_find=None,
                 scheduler='onecycle', scheduler_kwargs=None):
        super().__init__(model, dls, loss_func, opt_func, lr, cbs, init=init, lr_find=lr_find,
                         scheduler=scheduler, scheduler_kwargs=scheduler_kwargs)
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
    if kwargs.get('ex_inp') is None and kwargs.get('inp_size') is None:
        b = self.dls.train.one_batch()
        kwargs['ex_inp'] = b[0][:1]
    return describe_model(self.model, **kwargs)
