"""Tests for plain / Rich console tracking UI."""
import os

from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from odyssey.tracking.ui import (
    _PlainProgress,
    _compact_metric_line,
    _dict_metrics_table,
    _loss_sparklines,
    _metrics_table,
    _train_progress,
    is_plain,
    set_plain,
)
from odyssey.training.callbacks import ProgressCB


def setup_function():
    set_plain(None)
    os.environ.pop("ODYSSEY_PLAIN", None)


def teardown_function():
    set_plain(None)
    os.environ.pop("ODYSSEY_PLAIN", None)


def test_set_plain_override():
    set_plain(True)
    assert is_plain() is True
    set_plain(False)
    assert is_plain() is False
    set_plain(None)
    assert is_plain() is False


def test_env_plain(monkeypatch):
    monkeypatch.setenv("ODYSSEY_PLAIN", "1")
    set_plain(None)
    assert is_plain() is True
    monkeypatch.setenv("ODYSSEY_PLAIN", "true")
    assert is_plain() is True
    monkeypatch.setenv("ODYSSEY_PLAIN", "0")
    assert is_plain() is False


def test_plain_formatters_return_str():
    set_plain(True)
    d = {"epoch": "1", "loss": "0.5", "val_loss": "0.4", "accuracy": "0.9"}
    line = _compact_metric_line(d, n_epochs=10, best=True)
    assert isinstance(line, str)
    assert "epoch 1/10" in line
    assert "vloss 0.4000*" in line

    table = _dict_metrics_table({"epoch": "0", "train": "train", "loss": "0.1"})
    assert isinstance(table, str)
    assert "loss=0.1000" in table

    spark = _loss_sparklines([1.0, 0.5], [0.9, 0.4])
    assert isinstance(spark, str)
    assert spark.startswith("train ")

    prog = _train_progress()
    assert isinstance(prog, _PlainProgress)
    tid = prog.add_task("train", total=10)
    prog.update(tid, advance=3, description="train loss 0.1")
    assert "3/10" in prog.status_line()


def test_rich_formatters_when_not_plain():
    set_plain(False)
    line = _compact_metric_line({"epoch": "1", "loss": "0.5"}, n_epochs=5)
    assert isinstance(line, Text)
    assert isinstance(_dict_metrics_table({"loss": "0.1"}), Table)
    assert isinstance(_metrics_table(["epoch", "loss"]), Table)
    assert isinstance(_train_progress(), Progress)


def test_progress_cb_plain_skips_live():
    cb = ProgressCB(plain=True, min_refresh_interval=0)
    assert cb.plain is True

    class _Learn:
        epochs = range(1)

    learn = _Learn()
    cb.before_fit(learn)
    assert cb.live is None
    assert cb._nb_display is None
    assert isinstance(cb.progress, _PlainProgress)
    cb.after_fit(learn)
