"""Rich console helpers for training progress."""
import math
import os

import matplotlib.pyplot as plt
from rich.box import SIMPLE_HEAVY
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

_SPARK_CHARS = "▁▂▃▄▅▆▇█"
_METRIC_HISTORY = 5
_TOP_PARAMS = 8
_DESCRIBE_MAX_LAYERS = 20
_DESCRIBE_GROUP_DEPTH = 2
# Notebook Live refreshes are expensive; keep them sparse.
_NOTEBOOK_MIN_REFRESH = 1.0


def in_notebook() -> bool:
    "True in Jupyter / Colab."
    try:
        ipython = get_ipython()  # type: ignore[name-defined]
    except NameError:
        return False
    shell = ipython.__class__.__name__
    return (
        "google.colab" in str(ipython.__class__)
        or bool(os.getenv("DATABRICKS_RUNTIME_VERSION"))
        or shell == "ZMQInteractiveShell"
    )


def _make_console() -> Console:
    # Terminal: stderr keeps progress out of captured stdout (pipes/tests).
    # Notebook/Colab: plain stdout; progress uses NotebookDisplay, not Rich Live.
    if in_notebook():
        return Console(force_terminal=True, width=100)
    return Console(stderr=True)


_console = None


def console():
    global _console
    if _console is None:
        _console = _make_console()
    return _console


def notebook_min_refresh():
    return _NOTEBOOK_MIN_REFRESH


class NotebookDisplay:
    """Single in-place Jupyter/Colab output (no Rich Live, no cell flooding)."""

    def __init__(self):
        self._handle = None

    def update(self, renderable) -> None:
        from IPython.display import display
        from rich.jupyter import JupyterRenderable, _render_segments

        # Group has no JupyterMixin; render to HTML ourselves for display_id updates.
        c = console()
        segments = list(c.render(renderable, c.options))
        html = _render_segments(segments)
        text = c._render_buffer(segments)
        shim = JupyterRenderable(html, text)
        if self._handle is None:
            self._handle = display(shim, display_id=True)
        else:
            self._handle.update(shim)

    def close(self) -> None:
        self._handle = None


def metric_history():
    return _METRIC_HISTORY


def top_params():
    return _TOP_PARAMS


def describe_max_layers():
    return _DESCRIBE_MAX_LAYERS


def describe_group_depth():
    return _DESCRIBE_GROUP_DEPTH


def safe_plt_show(fig, show=False):
    "Show figure only when the matplotlib backend supports interactive display."
    if not show:
        plt.close(fig)
        return
    if plt.get_backend().lower().endswith("agg"):
        plt.close(fig)
        return
    plt.show()


def _finite_losses(data, max_pts=80):
    pts = [float(v) for v in (list(data[-max_pts:]) if data else [])]
    return [v for v in pts if math.isfinite(v)]

def _sparkline_str(data, max_pts=80):
    pts = _finite_losses(data, max_pts)
    if not pts: return _SPARK_CHARS[0] * min(len(data or []), 40) or _SPARK_CHARS[0]
    lo, hi = min(pts), max(pts)
    if hi == lo: return _SPARK_CHARS[len(_SPARK_CHARS) // 2] * min(len(pts), 40)
    n = len(_SPARK_CHARS) - 1
    return ''.join(_SPARK_CHARS[max(0, min(n, int((v - lo) / (hi - lo) * n)))] for v in pts)

def _train_progress():
    return Progress(
        TextColumn("[cyan]▸[/cyan]"),
        TextColumn("[bold]{task.description}"),
        BarColumn(bar_width=24, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console(),
        expand=False,
        disable=False,
    )

def _metric_style(key, val, *, best=False):
    if best and key in ('val_loss', 'loss'): return "bold green"
    if key in ('loss', 'val_loss'): return "yellow"
    if 'accuracy' in key: return "bold cyan"
    if 'dice' in key or 'iou' in key: return "bold cyan"
    if key in ('batch_ms', 'samp_s'): return "magenta"
    if key in ('epoch', 'val_epoch'): return "dim"
    return None

def _format_metric_cell(key, val):
    if key in ('epoch', 'val_epoch'): return str(val)
    if key in ('batch_ms',): return str(val)
    if key in ('samp_s',):
        try: return f'{float(val):.0f}'
        except (TypeError, ValueError): return str(val)
    try: return f'{float(val):.4f}'
    except (TypeError, ValueError): return str(val)

def _format_batch_time(sec):
    "Human-readable batch duration from seconds."
    ms = sec * 1000
    return f"{ms:.1f}ms" if ms < 1000 else f"{sec:.2f}s"

def _styled_metric(key, val, *, best=False):
    text = _format_metric_cell(key, val)
    style = _metric_style(key, val, best=best)
    return Text(text, style=style) if style else text

def _compact_metric_line(d, *, epoch=None, n_epochs=None, best=False):
    "Single-line colored summary: epoch · train metrics · val metrics."
    ep = d.get('epoch', epoch)
    parts = []
    if ep is not None:
        span = f"/{n_epochs}" if n_epochs else ""
        parts.append(Text.assemble(("epoch ", "dim"), (f"{ep}{span}", "bold white")))
    for k, label, color in (
        ('loss', 'loss', 'yellow'), ('accuracy', 'acc', 'cyan'),
        ('batch_ms', 'batch', 'magenta'), ('samp_s', 's/s', 'magenta'),
        ('val_loss', 'vloss', 'yellow'), ('val_accuracy', 'vacc', 'cyan'),
    ):
        if k not in d: continue
        tag = "bold green" if best and k == 'val_loss' else color
        parts.append(Text(" · "))
        parts.append(Text.assemble((f"{label} ", "dim"), (_format_metric_cell(k, d[k]), tag)))
    line = Text("")
    for p in parts: line.append_text(p)
    return line

def _metrics_table(columns):
    labels = {'accuracy': 'acc', 'val_accuracy': 'vacc', 'val_loss': 'vloss', 'val_epoch': 'vep',
              'batch_ms': 'batch', 'samp_s': 's/s'}
    t = Table(box=SIMPLE_HEAVY, show_header=True, header_style="bold cyan",
              padding=(0, 1), show_edge=False, expand=False)
    numeric = {'loss', 'val_loss', 'accuracy', 'val_accuracy', 'samp_s'}
    for col in columns:
        if col == 'train': continue
        t.add_column(labels.get(col, col), justify="right" if col in numeric or col.startswith('val_') else "left")
    return t

def _dict_metrics_table(log):
    t = Table(box=SIMPLE_HEAVY, show_header=True, header_style="bold cyan", padding=(0, 1), expand=False)
    for k in log:
        t.add_column(k, justify="right" if k not in ('epoch', 'train') else "left")
    t.add_row(*[_styled_metric(k, v) if k not in ('train',) else str(v) for k, v in log.items()])
    return t

def _loss_sparklines(train_losses, val_losses, max_pts=80):
    return Text.assemble(
        ("train ", "dim"), (_sparkline_str(train_losses, max_pts), "cyan"),
        ("  val ", "dim"), (_sparkline_str(val_losses, max_pts), "green"),
    )
