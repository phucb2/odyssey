"""Unit tests for Learner, grad accumulation, and default callback helpers."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from odyssey.training import (
    DeviceCB,
    GradAccumCB,
    GradClipCB,
    LRFind,
    Learner,
    MetricsCB,
    ProgressCB,
    TrainCB,
    default_cbs,
    make_lr_find,
)
from odyssey.training.config import AnalysisConfig


class _TinyDS(Dataset):
    def __init__(self, n: int = 8):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        return torch.randn(3, 8, 8), torch.tensor(i % 2)


class _TinyDls:
    def __init__(self, n: int = 8, batch_size: int = 2):
        self.train = DataLoader(_TinyDS(n), batch_size=batch_size)
        self.valid = DataLoader(_TinyDS(n), batch_size=batch_size)

    def one_batch(self):
        return next(iter(self.train))


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 8 * 8, 2)

    def forward(self, x):
        return self.fc(x.flatten(1))


class _StepCounterCB(TrainCB):
    def __init__(self):
        self.steps = 0

    def step(self, learn):
        if getattr(learn, "should_step", True):
            self.steps += 1
        super().step(learn)


def _cpu_learner(model=None, dls=None, cbs=None):
    cbs = cbs if cbs is not None else [TrainCB(), MetricsCB()]
    return Learner(
        model or _TinyModel(),
        dls or _TinyDls(),
        nn.CrossEntropyLoss(),
        torch.optim.SGD,
        lr=0.1,
        cbs=cbs,
        scheduler="none",
    )


def test_default_cbs_order():
    cbs = default_cbs(train=TrainCB(), channels_last=True, compile_warmup_batches=0)
    names = [type(cb).__name__ for cb in cbs]
    assert names.index("TimingCB") < names.index("MetricsCB")
    assert names.index("DeviceCB") < names.index("ProgressCB")
    assert names.index("MixPrecisionCB") < names.index("TrainCB")
    assert "ChannelsLastCB" in names
    assert isinstance(cbs[-1], TrainCB)


def test_default_cbs_grad_clip():
    cbs = default_cbs(train=TrainCB(), grad_clip_norm=1.0)
    assert isinstance(cbs[-1], GradClipCB)


def test_default_cbs_grad_accum():
    cbs = default_cbs(train=TrainCB(), grad_accum=4)
    assert any(isinstance(cb, GradAccumCB) for cb in cbs)
    assert isinstance(cbs[-1], TrainCB)


def test_auto_inject_train_cb():
    learn = _cpu_learner(cbs=[MetricsCB()])
    assert any(isinstance(cb, TrainCB) for cb in learn.cbs)


def test_auto_train_false_raises():
    with pytest.raises(ValueError, match="train callback"):
        Learner(
            _TinyModel(),
            _TinyDls(),
            nn.CrossEntropyLoss(),
            torch.optim.SGD,
            cbs=[MetricsCB()],
            scheduler="none",
            auto_train=False,
        )


def test_learner_fit_updates_weights():
    model = _TinyModel()
    before = model.fc.weight.detach().clone()
    learn = _cpu_learner(model=model)
    learn.fit(1)
    assert not torch.allclose(model.fc.weight, before)


def test_validate():
    learn = _cpu_learner()
    learn.validate()


def test_pin_memory_and_project_name():
    learn = Learner(
        _TinyModel(),
        _TinyDls(),
        nn.CrossEntropyLoss(),
        torch.optim.SGD,
        pin_memory=True,
        project_name="test-project",
        scheduler="none",
    )
    assert learn.pin_memory is True
    assert learn.project_name == "test-project"


def test_make_lr_find_from_analysis():
    analysis = AnalysisConfig(lr_epochs=2, show_plot=False)
    lr_find = make_lr_find(analysis)
    assert lr_find.n_epochs == 2
    assert lr_find.show_plot is False


def test_find_lr_smoke(monkeypatch):
    learn = _cpu_learner()
    called = {}

    def fake_run(_self, _learn):
        called["ran"] = True
        return None, None

    monkeypatch.setattr(LRFind, "run", fake_run)
    learn.find_lr(show_plot=False)
    assert called["ran"]


def test_training_public_exports():
    assert Learner is not None
    assert default_cbs is not None
    assert GradAccumCB is not None
    assert TrainCB is not None
    assert DeviceCB is not None
    assert ProgressCB is not None
    assert LRFind is not None


def test_grad_accum_step_frequency():
    counter = _StepCounterCB()
    learn = _cpu_learner(
        dls=_TinyDls(n=8, batch_size=2),
        cbs=[GradAccumCB(n_accum=2), counter, MetricsCB()],
    )
    learn.fit(1)
    assert counter.steps == 2


def test_grad_accum_steps_per_epoch():
    learn = _cpu_learner(
        dls=_TinyDls(n=10, batch_size=2),
        cbs=[GradAccumCB(n_accum=4), TrainCB()],
    )
    learn.fit(1)
    assert learn.steps_per_epoch == 2


def test_grad_accum_tail_flush():
    counter = _StepCounterCB()
    learn = _cpu_learner(
        dls=_TinyDls(n=5, batch_size=1),
        cbs=[GradAccumCB(n_accum=2), counter, MetricsCB()],
    )
    learn.fit(1)
    assert counter.steps == 3


def test_grad_accum_updates_weights():
    model = _TinyModel()
    before = model.fc.weight.detach().clone()
    learn = _cpu_learner(
        model=model,
        cbs=[GradAccumCB(n_accum=2), TrainCB(), MetricsCB()],
    )
    learn.fit(1)
    assert not torch.allclose(model.fc.weight, before)


def test_grad_accum_onecycle_scheduler_no_overflow():
    learn = Learner(
        _TinyModel(),
        _TinyDls(n=5, batch_size=1),
        nn.CrossEntropyLoss(),
        torch.optim.SGD,
        lr=0.1,
        cbs=[GradAccumCB(n_accum=2), TrainCB()],
        scheduler="onecycle",
    )
    learn.fit(2)
