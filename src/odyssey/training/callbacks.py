"""Training callbacks."""
import collections
import contextlib
import math
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path

import fastcore.all as fc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from fastai.callback.schedule import minimum, slide, steep, valley
from rich.panel import Panel
from rich.progress import Group
from rich.live import Live
from rich.text import Text
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import Mean

from odyssey.training.callback import (
    Callback,
    CancelBatchException,
    CancelEpochException,
    CancelFitException,
    to_cpu,
    to_device,
)
from odyssey.training.cuda import torch_compile_available
from odyssey.training.schedule import SchedulerCB
from odyssey.tracking.diagnostics import (
    _activation_hook_layers,
    _activation_log1p_abs_stats,
    plot_activation_hists,
    plot_dead_neurons,
)
from odyssey.models import ResBlock
from odyssey.paths import DEFAULT_PROJECT, run_path
from odyssey.tracking.ui import (
    NotebookDisplay,
    _PlainProgress,
    _compact_metric_line,
    _dict_metrics_table,
    _format_batch_time,
    _format_plain_metrics_table,
    _loss_sparklines,
    _metrics_table,
    _styled_metric,
    _train_progress,
    console,
    in_notebook,
    is_plain,
    metric_history,
    notebook_min_refresh,
    safe_plt_show,
)

_METRIC_HISTORY = metric_history()

class MetricsCB(Callback):
    def __init__(self, *ms, metric_interval=10, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()
        self.metric_interval = metric_interval
        self._batch_count = 0
    def before_fit(self, learn):
        learn.metrics = self
    def before_epoch(self, learn):
        self._batch_count = 0
        learn.epoch_metrics = {}
        [o.reset() for o in self.all_metrics.values()]
    def after_epoch(self, learn):
        log={k:f'{v.compute():.4f}' for k,v in self.all_metrics.items()}
        log.update(getattr(learn, "epoch_metrics", {}))
        log['epoch'] = str(learn.epoch)
        log['train'] = 'train' if learn.model.training else 'valid'
        self._log(log)
        
    def after_batch(self, learn):
        bs = len(learn.batch[0]) if learn.batch else 0
        self.loss.update(to_cpu(learn.loss), weight=bs)
        update_metrics = not learn.training or (self._batch_count % self.metric_interval == 0)
        self._batch_count += 1
        if not update_metrics or not self.metrics:
            return
        if isinstance(learn.preds, dict):
            return
        metric_y = getattr(learn, 'batch_labels', learn.batch[1])
        if isinstance(metric_y, torch.Tensor) and metric_y.ndim > 1:
            # [B, 1] class ids (MedMNIST) are not one-hot; argmax would always be 0.
            metric_y = metric_y.squeeze(-1) if metric_y.shape[-1] == 1 else metric_y.argmax(dim=-1)
        for m in self.metrics.values(): m.update(to_cpu(learn.preds), to_cpu(metric_y))
    def _log(self, log):
        console().print(_dict_metrics_table(log))


class LossDictMetricsCB(Callback):
    """Track named components from `learn.loss_dict` into `learn.epoch_metrics`."""

    order = MetricsCB.order - 1

    def __init__(self, keys: tuple[str, ...]):
        self.keys = keys
        self._means: dict[str, Mean] = {k: Mean() for k in keys}

    def before_epoch(self, learn):
        for mean in self._means.values():
            mean.reset()

    def after_batch(self, learn):
        loss_dict = getattr(learn, "loss_dict", None)
        if not loss_dict:
            return
        bs = len(learn.batch[0]) if learn.batch else 0
        for key in self.keys:
            if key in loss_dict:
                self._means[key].update(to_cpu(loss_dict[key]), weight=bs)

    def after_epoch(self, learn):
        learn.epoch_metrics = getattr(learn, "epoch_metrics", {})
        for key, mean in self._means.items():
            learn.epoch_metrics[key] = f"{mean.compute():.4f}"


default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DeviceCB(Callback):
    order = -4
    def __init__(self, device=default_device): 
        fc.store_attr()
    def before_fit(self, learn): 
        learn.device = self.device
        if hasattr(learn.model, 'to'): learn.model.to(self.device)
        learn.pin_memory = bool(getattr(learn, 'pin_memory', False))
        learn.non_blocking = bool(
            getattr(learn.device, 'type', None) == 'cuda' and learn.pin_memory
        )
    def before_batch(self, learn):
        x = learn.batch[0]
        if isinstance(x, torch.Tensor) and x.device == self.device:
            return
        learn.batch = to_device(learn.batch, self.device, bool(getattr(learn, "non_blocking", False)))

class CompileCB(Callback):
    "Wrap model with torch.compile before fit (CUDA + Triton only)."
    order = -2
    def __init__(self, mode="default", enabled=True):
        self.mode = mode
        self.enabled = enabled
    def before_fit(self, learn):
        if not self.enabled: return
        device = getattr(learn, 'device', default_device)
        if getattr(device, 'type', None) != 'cuda': return
        if not torch_compile_available():
            warnings.warn(
                "compile=True but Triton is unavailable (common on Windows). "
                "Skipping torch.compile; pass compile=false to silence this.",
                stacklevel=2,
            )
            return
        learn.model = torch.compile(learn.model, mode=self.mode)


def _is_compiled_model(model):
    return getattr(model, '_orig_mod', None) is not None


def _prepare_learn_batch(learn, batch):
    batch = to_device(batch, learn.device, bool(getattr(learn, "non_blocking", False)))
    if getattr(learn, 'use_channels_last', False):
        x, *rest = batch
        if isinstance(x, torch.Tensor):
            batch = (x.to(memory_format=torch.channels_last), *rest)
    return batch


class CompileWarmupCB(Callback):
    "Run a few train steps after compile so epoch 1 is not spent warming the engine."
    order = 1

    def __init__(self, n_batches=4):
        self.n_batches = n_batches

    def before_fit(self, learn):
        if self.n_batches <= 0 or not _is_compiled_model(learn.model):
            return
        device = getattr(learn, 'device', default_device)
        if getattr(device, 'type', None) != 'cuda':
            return

        was_training = learn.model.training
        learn.model.train()
        saved = {k: v.detach().clone() for k, v in learn.model.state_dict().items()}
        batch = _prepare_learn_batch(learn, next(iter(learn.dls.train)))

        # one_batch() skips before_batch hooks (AMP autocast). Wrap here when fp16 data + AMP.
        # Detach LR schedulers so warmup opt steps do not consume OneCycle / cosine budgets.
        use_amp = getattr(learn, '_use_amp', False)
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        held_sched = [cb for cb in learn.cbs if isinstance(cb, SchedulerCB)]
        if held_sched:
            learn.cbs = [cb for cb in learn.cbs if not isinstance(cb, SchedulerCB)]
        try:
            with amp_ctx:
                for _ in range(self.n_batches):
                    learn.batch = batch
                    learn.one_batch()
        finally:
            if held_sched:
                learn.cbs = list(learn.cbs) + held_sched

        learn.model.load_state_dict(saved)
        learn.model.train(was_training)
        if getattr(learn, 'opt', None) is not None:
            learn.opt.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

def _model_uses_conv(model):
    for mod in model.modules():
        if isinstance(mod, (nn.Conv2d, ResBlock)):
            return True
    return False

class ChannelsLastCB(Callback):
    "Use channels_last memory format for CNN models on CUDA."
    order = -3
    def before_fit(self, learn):
        device = getattr(learn, 'device', default_device)
        self.use_channels_last = (
            getattr(device, 'type', None) == 'cuda' and _model_uses_conv(learn.model)
        )
        if self.use_channels_last:
            learn.model = learn.model.to(memory_format=torch.channels_last)
        learn.use_channels_last = self.use_channels_last
    def before_batch(self, learn):
        if not self.use_channels_last or learn.batch is None: return
        x = learn.batch[0]
        if isinstance(x, torch.Tensor):
            learn.batch = (x.to(memory_format=torch.channels_last), *learn.batch[1:])

class TimingCB(Callback):
    "Per-batch wall time for performance monitoring; omit from `cbs` to disable."
    order = 0
    def __init__(self, *, sync: bool = False):
        self.sync = sync
        self._t0 = None
        self._epoch_times = []

    def before_fit(self, learn):
        learn.batch_time = None
        learn.train_batch_time = learn.train_samples_per_sec = None

    def _sync(self, learn):
        if not self.sync: return
        dev = getattr(learn, 'device', None)
        if dev is not None and getattr(dev, 'type', None) == 'cuda':
            torch.cuda.synchronize(dev)

    def before_epoch(self, learn):
        self._epoch_times = [] if learn.training else self._epoch_times

    def before_batch(self, learn):
        self._sync(learn)
        self._t0 = time.perf_counter()

    def after_batch(self, learn):
        self._sync(learn)
        dt = time.perf_counter() - self._t0
        learn.batch_time = dt
        bs = len(learn.batch[0]) if learn.batch else 0
        learn.batch_samples_per_sec = bs / dt if dt > 0 else 0.0
        if learn.training: self._epoch_times.append(dt)

    def after_epoch(self, learn):
        if not learn.training or not self._epoch_times: return
        avg = sum(self._epoch_times) / len(self._epoch_times)
        bs = len(learn.batch[0]) if learn.batch else 0
        learn.train_batch_time = avg
        learn.train_samples_per_sec = bs / avg if avg > 0 else 0.0

def clip_gradients(params, *, max_norm=None, max_value=None):
    "Clip gradients in-place; pass `max_norm` and/or `max_value` (at least one required)."
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    if max_norm is not None:
        nn.utils.clip_grad_norm_(grads, max_norm)
    if max_value is not None:
        nn.utils.clip_grad_value_(grads, max_value)

class MixPrecisionCB(Callback):
    "CUDA AMP via callback hooks; composes with GradAccumCB and GradClipCB."

    order = -1

    def __init__(self):
        self._enabled = False
        self._autocast = None

    def before_fit(self, learn):
        device = getattr(learn, "device", default_device)
        self._enabled = getattr(device, "type", None) == "cuda"
        if self._enabled:
            learn._use_amp = True
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)

    def before_batch(self, learn):
        # Autocast on train and eval: GPU data is often cached as fp16 while weights stay fp32.
        if self._enabled:
            self._autocast = torch.autocast("cuda", dtype=torch.float16)
            self._autocast.__enter__()
        else:
            self._autocast = None

    def after_batch(self, learn):
        if self._autocast is not None:
            self._autocast.__exit__(None, None, None)
            self._autocast = None

    def backward(self, learn):
        if not self._enabled:
            return
        self.scaler.scale(learn.loss).backward()

    def step(self, learn):
        if not self._enabled or not getattr(learn, "should_step", True):
            return
        self.scaler.unscale_(learn.opt)
        learn.clip_grad()
        self.scaler.step(learn.opt)
        self.scaler.update()

    def zero_grad(self, learn):
        if not self._enabled or not getattr(learn, "should_step", True):
            return
        learn.opt.zero_grad(set_to_none=True)


class GradAccumCB(Callback):
    "Accumulate gradients over micro-batches; step every `n_accum` train batches."

    order = -1

    def __init__(self, n_accum=2, *, scale_loss=True, step_on_epoch_end=True):
        if n_accum < 1:
            raise ValueError(f"n_accum must be >= 1, got {n_accum}")
        fc.store_attr()

    def before_fit(self, learn):
        learn.n_accum = self.n_accum
        learn.accum_step = 0
        learn.should_step = self.n_accum == 1

    def before_epoch(self, learn):
        learn.accum_step = 0
        learn.should_step = self.n_accum == 1

    def before_backward(self, learn):
        if self.scale_loss and self.n_accum > 1:
            learn.loss = learn.loss / self.n_accum

    def after_backward(self, learn):
        learn.accum_step += 1
        learn.should_step = learn.accum_step >= self.n_accum

    def after_batch(self, learn):
        if not learn.training or not learn.should_step:
            return
        learn.accum_step = 0
        learn.should_step = False

    def after_epoch(self, learn):
        if not learn.training or not self.step_on_epoch_end or learn.accum_step <= 0:
            return
        learn.should_step = True
        learn._optimizer_step()
        learn.accum_step = 0
        learn.should_step = False


class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False, min_refresh_interval=None, plain=None):
        self.plot = plot
        self.plain = is_plain() if plain is None else bool(plain)
        # Colab/Jupyter: slower refresh so the single display handle stays cheap.
        if min_refresh_interval is None:
            min_refresh_interval = notebook_min_refresh() if in_notebook() else 0.25
        self.min_refresh_interval = min_refresh_interval

    def before_fit(self, learn):
        self.learn = learn
        self.n_epochs = len(learn.epochs) if hasattr(learn.epochs, '__len__') else None
        self.console = console()
        self.progress = _train_progress(plain=self.plain)
        self.epoch_task = self.progress.add_task("epochs", total=self.n_epochs)
        self.batch_task = None
        self.metrics_table = None
        self.columns = None
        self.metric_rows = []
        self.summary = "" if self.plain else Text("")
        self.latest_is_best = False
        self.first = True
        self.losses, self.val_losses = [], []
        self.best_val_loss = math.inf
        self._last_refresh = 0.0
        self._pending_refresh = False
        self._notebook = in_notebook()
        self.live = None
        self._nb_display = None
        if self.plain:
            pass  # text-only: print on refresh, no Live / NotebookDisplay
        elif self._notebook:
            # DisplayHandle updates one output in place — Rich Live appends in Colab.
            self._nb_display = NotebookDisplay()
            self._nb_display.update(self._group())
        else:
            self.live = Live(
                self._group(), console=self.console,
                auto_refresh=False,
                vertical_overflow="visible",
            )
            self.live.start()
        if getattr(learn, 'metrics', None) is not None:
            learn.metrics._log = self._log

    def _group(self):
        items = [self.progress, Panel(self.summary, border_style="green" if self.latest_is_best else "cyan", padding=(0, 1))]
        if self.metrics_table is not None: items.append(self.metrics_table)
        if self.plot: items.append(_loss_sparklines(self.losses, self.val_losses, plain=False))
        return Group(*items)

    def _plain_block(self, *, full=True) -> str:
        lines = []
        if isinstance(self.progress, _PlainProgress):
            status = self.progress.status_line()
            if status:
                lines.append(status)
        if not full:
            return "\n".join(lines)
        if self.summary:
            lines.append(str(self.summary))
        if self.metrics_table is not None:
            lines.append(_format_plain_metrics_table(self.metrics_table))
        if self.plot:
            lines.append(_loss_sparklines(self.losses, self.val_losses, plain=True))
        return "\n".join(lines)

    def _rebuild_metrics_table(self):
        if not self.columns: return
        self.metrics_table = _metrics_table(self.columns, plain=self.plain)
        for i, d in enumerate(self.metric_rows):
            is_best = i == len(self.metric_rows) - 1 and float(d.get('val_loss', math.inf)) <= self.best_val_loss
            cells = [
                _styled_metric(k, d[k], best=is_best and k == 'val_loss', plain=self.plain)
                for k in self.columns
            ]
            if self.plain:
                self.metrics_table["rows"].append(cells)
            else:
                self.metrics_table.add_row(*cells)

    def _should_refresh(self, *, force=False):
        if force: return True
        now = time.monotonic()
        return (now - self._last_refresh) >= self.min_refresh_interval

    def _refresh(self, *, force=False):
        now = time.monotonic()
        if not force and (now - self._last_refresh) < self.min_refresh_interval:
            self._pending_refresh = True
            return
        self._last_refresh = now
        self._pending_refresh = False
        if self.plain:
            block = self._plain_block(full=force)
            if block:
                self.console.print(block)
            return
        renderable = self._group()
        if self._nb_display is not None:
            self._nb_display.update(renderable)
        elif self.live is not None:
            self.live.update(renderable, refresh=True)

    def _stop_live(self):
        if self._pending_refresh: self._refresh(force=True)
        if self.plain:
            self.progress = None
            return
        if self._nb_display is not None:
            # Leave the last display_id frame visible; do not reprint.
            self._nb_display.close()
            self._nb_display = None
        if self.live is not None:
            # transient clear + one final print avoids Live.stop()'s second full frame
            # when the terminal can't restore the cursor (looks like duplicated logs).
            final = self._group()
            self.live.transient = True
            self.live.stop()
            self.live = None
            self.console.print(final)
        self.progress = None

    def after_fit(self, learn): self._stop_live()

    def _pivot_log(self, d):
        if d['train'] == 'train':
            self.train_log = d
            return None
        val_log = {f'val_{k}': v for k, v in d.items() if k != 'train'}
        merged = {**self.train_log, **val_log}
        l = getattr(self, 'learn', None)
        if l is not None and getattr(l, 'train_batch_time', None) is not None:
            merged['batch_ms'] = _format_batch_time(l.train_batch_time)
            merged['samp_s'] = f"{getattr(l, 'train_samples_per_sec', 0):.0f}"
        return merged

    def _log(self, d):
        d = self._pivot_log(d)
        if d is None: return
        if self.first:
            self.columns = [k for k in d if k != 'train']
            self.first = False
        val_loss = float(d.get('val_loss', math.inf))
        is_best = val_loss < self.best_val_loss
        if is_best: self.best_val_loss = val_loss
        self.latest_is_best = is_best
        self.metric_rows.append(d)
        if len(self.metric_rows) > _METRIC_HISTORY:
            self.metric_rows = self.metric_rows[-_METRIC_HISTORY:]
        self.summary = _compact_metric_line(d, n_epochs=self.n_epochs, best=is_best, plain=self.plain)
        self._rebuild_metrics_table()
        self._refresh(force=True)

    def before_epoch(self, learn):
        phase = 'train' if learn.training else 'valid'
        total = len(learn.dl)
        if self.batch_task is None:
            self.batch_task = self.progress.add_task(phase, total=total)
        else:
            self.progress.reset(self.batch_task, total=total, completed=0, visible=True, description=phase)

    def _batch_desc(self, learn, loss):
        phase = 'train' if learn.training else 'valid'
        loss_s = f"{loss:.3f}" if math.isfinite(loss) else "nan"
        desc = f"{phase} loss {loss_s}"
        dt = getattr(learn, 'batch_time', None)
        if dt is not None:
            desc += f" · {_format_batch_time(dt)}"
            sps = getattr(learn, 'batch_samples_per_sec', None)
            if sps: desc += f" · {sps:.0f} s/s"
        return desc

    def after_batch(self, learn):
        if self._should_refresh() or self._pending_refresh:
            loss = learn.loss.detach().float().item() if hasattr(learn.loss, 'item') else float(learn.loss)
        else:
            loss = getattr(self, '_last_loss', float('nan'))
        self._last_loss = loss
        self.progress.update(self.batch_task, advance=1, description=self._batch_desc(learn, loss))
        if self.plot and learn.training and math.isfinite(loss):
            self.losses.append(loss)
        self._refresh()

    def after_epoch(self, learn):
        if not learn.training:
            if self.plot and hasattr(learn, 'metrics'):
                self.val_losses.append(learn.metrics.all_metrics['loss'].compute())
            self.progress.update(self.epoch_task, advance=1, description=f"epoch {learn.epoch + 1}/{self.n_epochs}")
            self._refresh(force=True)

def _float_metrics(metrics):
    return {k: float(v.compute()) for k, v in metrics.items()}

class TensorBoardCB(Callback):
    "Log train/val metrics to TensorBoard; requires MetricsCB in cbs."
    order = ProgressCB.order + 1
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
    def before_fit(self, learn):
        project = getattr(learn, "project_name", DEFAULT_PROJECT)
        log_dir = self.log_dir or str(run_path(project, "tensorboard", datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.writer = SummaryWriter(log_dir)
        self._train, self._step = {}, 0
    def after_batch(self, learn):
        if not learn.training: return
        self.writer.add_scalar("train/loss", float(to_cpu(learn.loss)), self._step)
        if getattr(learn, 'opt', None):
            self.writer.add_scalar("lr", learn.opt.param_groups[0]['lr'], self._step)
        self._step += 1
    def after_epoch(self, learn):
        if not hasattr(learn, 'metrics'): return
        if learn.training:
            self._train = _float_metrics(learn.metrics.all_metrics)
            if getattr(learn, 'train_batch_time', None) is not None:
                self._train['batch_time'] = learn.train_batch_time
                self._train['samples_per_sec'] = learn.train_samples_per_sec
        else:
            for k, v in self._train.items():
                self.writer.add_scalar(f"train/{k}", v, learn.epoch)
            for k, v in _float_metrics(learn.metrics.all_metrics).items():
                self.writer.add_scalar(f"val/{k}", v, learn.epoch)
    def after_fit(self, learn):
        self.writer.close()

class WandbCB(Callback):
    "Log train/val metrics to Weights & Biases; requires MetricsCB in cbs."
    order = ProgressCB.order + 1
    def __init__(self, project=None, name=None, config=None):
        fc.store_attr()
    def before_fit(self, learn):
        import wandb
        self.wandb = wandb
        wandb.init(project=self.project, name=self.name, config=self.config)
        self._train, self._step = {}, 0
    def after_batch(self, learn):
        if not learn.training: return
        d = {"train/loss": float(to_cpu(learn.loss))}
        if getattr(learn, 'opt', None):
            d["lr"] = learn.opt.param_groups[0]['lr']
        self.wandb.log(d, step=self._step)
        self._step += 1
    def after_epoch(self, learn):
        if not hasattr(learn, 'metrics'): return
        if learn.training:
            self._train = _float_metrics(learn.metrics.all_metrics)
            if getattr(learn, 'train_batch_time', None) is not None:
                self._train['batch_time'] = learn.train_batch_time
                self._train['samples_per_sec'] = learn.train_samples_per_sec
        else:
            d = {f"train/{k}": v for k, v in self._train.items()}
            d.update({f"val/{k}": v for k, v in _float_metrics(learn.metrics.all_metrics).items()})
            self.wandb.log(d, step=learn.epoch)
    def after_fit(self, learn):
        self.wandb.finish()

class LRFinder(Callback):
    "Exponential LR sweep; stops when loss diverges (fastai-style 4× min by default)."
    order = ProgressCB.order+1
    def __init__(self, lr_mult=1.3, stop_div_factor=4.0):
        self.lr_mult = lr_mult
        self.stop_div_factor = stop_div_factor
        
    def before_fit(self, learn):
        console().print("[dim]LRFinder: scanning learning rates…[/dim]")
        self.lrs,self.losses = [],[]
        self.min = math.inf
        
    def after_batch(self, learn):
        if not learn.training: raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = float(to_cpu(learn.loss))
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if loss > self.min * self.stop_div_factor:
            console().print(f"[yellow]LRFinder: loss > {self.stop_div_factor:.0f}× min — stopping[/yellow]")
            raise CancelFitException()
        for g in learn.opt.param_groups: g['lr'] *= self.lr_mult

_DEFAULT_SUGGEST_FUNCS = (valley, steep, minimum, slide)
_MIN_LR_POINTS = {"slide": 20, "minimum": 8, "valley": 6, "steep": 6}

def _trim_lr_find(lrs, losses, skip_start=0, skip_end=5):
    lrs, losses = list(lrs), list(losses)
    if skip_start: lrs, losses = lrs[skip_start:], losses[skip_start:]
    if skip_end: lrs, losses = lrs[:-skip_end] or lrs, losses[:-skip_end] or losses
    return lrs, losses

def _suggest_lrs(lrs, losses, suggest_funcs=_DEFAULT_SUGGEST_FUNCS):
    lrs_t = torch.tensor(lrs, dtype=torch.float64)
    losses_t = torch.tensor(losses, dtype=torch.float64)
    out, points, names = {}, {}, []
    n = len(lrs)
    for func in suggest_funcs:
        name = func.__name__
        if n < _MIN_LR_POINTS.get(name, 6):
            continue
        lr, (lr_pt, loss_pt) = func(lrs_t, losses_t, n)
        out[name] = float(lr)
        points[name] = (float(lr_pt), float(loss_pt))
        names.append(name)
    if not names:
        raise RuntimeError("Not enough LR finder points for suggestions")
    return collections.namedtuple('SuggestedLRs', names)(*out.values()), points, names

def plot_lr_find(lrs, losses, suggestions=None, names=None, *, title="Learning rate finder",
                 save_path=None, show=True):
    "Log-scale loss vs LR with optional fastai suggestion markers."
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale("log")
    ax.set_title(title)
    if suggestions:
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]
        for (lr, loss), nm, color in zip(suggestions.values(), names, colors):
            ax.plot(lr, loss, "o", label=f"{nm} ({lr:.2e})", c=color)
        ax.legend(loc="best")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        console().print(f"[green]LR finder plot saved → {save_path}[/green]")
    _safe_plt_show(fig, show)
    return fig

def _report_lr_suggestions(suggested, names):
    summary = Text("")
    styles = {"valley": "bold green", "steep": "cyan", "minimum": "yellow", "slide": "magenta"}
    for i, nm in enumerate(names):
        if i: summary.append_text(Text(" · "))
        summary.append_text(Text.assemble(
            (f"{nm} ", "dim"), (f"{getattr(suggested, nm):.2e}", styles.get(nm, "white")),
        ))
    console().print(Panel(summary, title="[bold]Suggested learning rates[/bold]", border_style="green"))

class LRFind:
    "Standalone LR sweep; compose with Learner via `lr_find=` or call `run(learn)` / `lr_find(learn)`."
    def __init__(self, n_epochs=1, lr_mult=1.3, start_lr=None, show_plot=True, save_path=None,
                 suggest_funcs=_DEFAULT_SUGGEST_FUNCS, skip_start=0, skip_end=5, stop_div_factor=4.0):
        fc.store_attr()
        self.finder = None
        self.suggested = None
        self.lrs = None
        self.losses = None

    def run(self, learn):
        "Sweep LRs on `learn`, restore weights, plot/report suggestions."
        finder = LRFinder(lr_mult=self.lr_mult, stop_div_factor=self.stop_div_factor)
        saved_lr = learn.lr
        if self.start_lr is not None: learn.lr = self.start_lr
        state = deepcopy(learn.model.state_dict())
        orig_cbs = learn.cbs
        try:
            learn.cbs = orig_cbs + [finder]
            learn.fit(self.n_epochs)
        finally:
            learn.cbs = orig_cbs
            learn.model.load_state_dict(state)
            learn.lr = saved_lr
            if getattr(learn, 'opt', None) is not None:
                learn.opt.zero_grad(set_to_none=True)

        lrs, losses = _trim_lr_find(finder.lrs, finder.losses, self.skip_start, self.skip_end)
        if len(lrs) < 6:
            raise RuntimeError(f"LR finder stopped early ({len(lrs)} points); try lower lr_mult or more epochs")

        suggested, points, names = _suggest_lrs(lrs, losses, self.suggest_funcs)
        if self.show_plot or self.save_path:
            plot_lr_find(lrs, losses, suggestions=points, names=names,
                         save_path=self.save_path, show=self.show_plot)
        _report_lr_suggestions(suggested, names)

        self.finder, self.suggested, self.lrs, self.losses = finder, suggested, lrs, losses
        return suggested, finder

    __call__ = run


def make_lr_find(
    analysis=None,
    *,
    n_epochs: int = 1,
    lr_mult: float = 1.3,
    start_lr: float | None = None,
    show_plot: bool = True,
    save_path: str | None = None,
) -> LRFind:
    "Build LRFind from AnalysisConfig or explicit kwargs."
    if analysis is not None:
        return LRFind(
            n_epochs=analysis.lr_epochs,
            lr_mult=analysis.lr_mult,
            start_lr=analysis.lr,
            show_plot=analysis.show_plot,
            save_path=analysis.lr_save,
        )
    return LRFind(
        n_epochs=n_epochs,
        lr_mult=lr_mult,
        start_lr=start_lr,
        show_plot=show_plot,
        save_path=save_path,
    )


def default_cbs(
    *,
    train: Callback,
    metrics: dict | None = None,
    compile: bool = False,
    compile_mode: str = "default",
    compile_warmup_batches: int | None = None,
    include_compile_cb: bool = True,
    channels_last: bool = False,
    plot_progress: bool = True,
    grad_clip_norm: float | None = None,
    grad_clip_value: float | None = None,
    grad_accum: int = 1,
    before_metrics: list | None = None,
    after_device: list | None = None,
) -> list:
    "Standard callback stack shared by classification, DETR, and CLIP builders."
    cbs: list = []
    if include_compile_cb:
        cbs.append(CompileCB(mode=compile_mode, enabled=compile))
    cbs.append(TimingCB())
    if before_metrics:
        cbs.extend(before_metrics)
    cbs.append(MetricsCB(**(metrics or {})))
    cbs.append(DeviceCB())
    if after_device:
        cbs.extend(after_device)
    if channels_last:
        cbs.append(ChannelsLastCB())
    cbs.append(ProgressCB(plot=plot_progress))
    cbs.append(MixPrecisionCB())
    if compile_warmup_batches is not None:
        cbs.append(CompileWarmupCB(n_batches=compile_warmup_batches))
    if grad_accum > 1:
        cbs.append(GradAccumCB(n_accum=grad_accum))
    cbs.append(train)
    if grad_clip_norm is not None or grad_clip_value is not None:
        cbs.append(GradClipCB(max_norm=grad_clip_norm, max_value=grad_clip_value))
    return cbs


class TrainCB(Callback):
    def predict(self, learn):
        return learn.model(learn.batch[0])

    def get_loss(self, learn):
        return learn.loss_func(learn.preds, learn.batch[1])

    def backward(self, learn):
        if getattr(learn, "_use_amp", False):
            return
        learn.loss.backward()

    def step(self, learn):
        if not getattr(learn, "should_step", True):
            return
        if getattr(learn, "_use_amp", False):
            return
        learn.opt.step()

    def zero_grad(self, learn):
        if not getattr(learn, "should_step", True):
            return
        if getattr(learn, "_use_amp", False):
            return
        learn.opt.zero_grad(set_to_none=True)


class GradClipCB(Callback):
    "Clip gradients after backward; use with MixPrecisionCB (unscale runs before this hook)."
    order = TrainCB.order

    def __init__(self, max_norm=None, max_value=None):
        if max_norm is None and max_value is None:
            raise ValueError("GradClipCB requires max_norm and/or max_value")
        self.max_norm = max_norm
        self.max_value = max_value

    def clip_grad(self, learn):
        if not getattr(learn, "should_step", True):
            return
        clip_gradients(learn.model.parameters(), max_norm=self.max_norm, max_value=self.max_value)


class ActivationHistCB(Callback):
    "Capture layer activations each train epoch; save compact epoch×bin heatmap; omit from `cbs` to disable."
    order = ProgressCB.order + 1
    def __init__(self, n_batches=2, n_bins=96, dead_eps=1e-6, save_dir=None, show=False):
        fc.store_attr()
        self.save_dir = Path(save_dir) if save_dir is not None else run_path(DEFAULT_PROJECT, "activations")
        self._hooks = []
        self._history = {}
        self._capturing = False
        self._batch_count = 0
        self._bufs = {}

    def before_fit(self, learn):
        self._history = {}
        self._register_hooks(learn.model)

    def after_fit(self, learn):
        self._remove_hooks()
        self._save_plots("Activation heatmap — final")

    def _register_hooks(self, model):
        self._remove_hooks()
        for name, mod in _activation_hook_layers(model):
            self._hooks.append(mod.register_forward_hook(self._hook(name)))

    def _remove_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

    def _hook(self, name):
        def _f(mod, inp, out):
            if not self._capturing: return
            t = out[0] if isinstance(out, tuple) else out
            if isinstance(t, torch.Tensor):
                self._bufs.setdefault(name, []).append(t.detach().float().cpu())
        return _f

    def before_epoch(self, learn):
        if not learn.training: return
        self._capturing = True
        self._batch_count = 0
        self._bufs = {}

    def after_batch(self, learn):
        if not learn.training or not self._capturing: return
        self._batch_count += 1
        if self._batch_count >= self.n_batches:
            self._capturing = False

    def after_epoch(self, learn):
        if not learn.training: return
        self._capturing = False
        for name, tensors in self._bufs.items():
            stats = _activation_log1p_abs_stats(tensors, self.n_bins, dead_eps=self.dead_eps)
            if stats is None: continue
            self._history.setdefault(name, []).append((learn.epoch, stats))
        self._save_plots(f"Activation heatmap — through epoch {learn.epoch}")

    def _save_plots(self, title):
        if not self._history: return
        plot_activation_hists(
            self._history,
            n_bins=self.n_bins,
            save_path=self.save_dir / "activations.png",
            show=self.show,
            title=title,
        )
        plot_dead_neurons(
            self._history,
            save_path=self.save_dir / "dead_neurons.png",
            show=self.show,
            title="Dead neurons per layer",
        )
