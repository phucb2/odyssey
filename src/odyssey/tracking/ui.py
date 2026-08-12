"""Console helpers for training progress (Rich or plain text)."""
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

# None → follow ODYSSEY_PLAIN; True/False → forced.
_plain_override = None
_console = None


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


def _env_plain() -> bool:
    return os.getenv("ODYSSEY_PLAIN", "").strip().lower() in {"1", "true", "yes", "on"}


def is_plain() -> bool:
    "True when console output should be text-only (no Rich Live/colors)."
    if _plain_override is not None:
        return _plain_override
    return _env_plain()


def set_plain(enabled: bool | None) -> None:
    """Force plain mode (`True`/`False`), or `None` to follow `ODYSSEY_PLAIN`."""
    global _plain_override, _console
    _plain_override = enabled
    _console = None


def _make_console() -> Console:
    # Terminal: stderr keeps progress out of captured stdout (pipes/tests).
    # Notebook/Colab: plain stdout; progress uses NotebookDisplay, not Rich Live.
    plain = is_plain()
    if in_notebook():
        return Console(
            force_terminal=not plain,
            width=100,
            no_color=plain,
            highlight=not plain,
        )
    return Console(
        stderr=True,
        no_color=plain,
        highlight=not plain,
        force_terminal=not plain,
    )


def console():
    global _console
    want_plain = is_plain()
    if _console is None or getattr(_console, "_odyssey_plain", None) != want_plain:
        _console = _make_console()
        _console._odyssey_plain = want_plain
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


class _PlainProgress:
    """Minimal progress tracker for plain console mode (no Rich Live/bars)."""

    def __init__(self):
        self._tasks = {}
        self._next_id = 0

    def add_task(self, description, total=None):
        tid = self._next_id
        self._next_id += 1
        self._tasks[tid] = {
            "description": description,
            "total": total,
            "completed": 0,
            "visible": True,
        }
        return tid

    def reset(self, task_id, total=None, completed=0, visible=True, description=None):
        t = self._tasks[task_id]
        if total is not None:
            t["total"] = total
        t["completed"] = completed
        t["visible"] = visible
        if description is not None:
            t["description"] = description

    def update(self, task_id, advance=0, description=None, completed=None):
        t = self._tasks[task_id]
        if completed is not None:
            t["completed"] = completed
        else:
            t["completed"] = t["completed"] + advance
        if description is not None:
            t["description"] = description

    def status_line(self) -> str:
        parts = []
        for t in self._tasks.values():
            if not t["visible"]:
                continue
            tot = t["total"]
            if tot is not None:
                parts.append(f"{t['description']} {t['completed']}/{tot}")
            else:
                parts.append(f"{t['description']} {t['completed']}")
        return " | ".join(parts)


def _train_progress(*, plain=None):
    if plain if plain is not None else is_plain():
        return _PlainProgress()
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

def _styled_metric(key, val, *, best=False, plain=None):
    text = _format_metric_cell(key, val)
    if plain if plain is not None else is_plain():
        return text
    style = _metric_style(key, val, best=best)
    return Text(text, style=style) if style else text

def _compact_metric_line(d, *, epoch=None, n_epochs=None, best=False, plain=None):
    "Single-line summary: epoch · train metrics · val metrics."
    use_plain = plain if plain is not None else is_plain()
    ep = d.get('epoch', epoch)
    if use_plain:
        parts = []
        if ep is not None:
            span = f"/{n_epochs}" if n_epochs else ""
            parts.append(f"epoch {ep}{span}")
        for k, label in (
            ('loss', 'loss'), ('accuracy', 'acc'),
            ('batch_ms', 'batch'), ('samp_s', 's/s'),
            ('val_loss', 'vloss'), ('val_accuracy', 'vacc'),
        ):
            if k not in d:
                continue
            mark = "*" if best and k == 'val_loss' else ""
            parts.append(f"{label} {_format_metric_cell(k, d[k])}{mark}")
        return " · ".join(parts)

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

_METRIC_LABELS = {
    'accuracy': 'acc', 'val_accuracy': 'vacc', 'val_loss': 'vloss', 'val_epoch': 'vep',
    'batch_ms': 'batch', 'samp_s': 's/s',
}


def _metrics_table(columns, *, plain=None):
    use_plain = plain if plain is not None else is_plain()
    cols = [c for c in columns if c != 'train']
    if use_plain:
        return {"columns": cols, "rows": []}

    t = Table(box=SIMPLE_HEAVY, show_header=True, header_style="bold cyan",
              padding=(0, 1), show_edge=False, expand=False)
    numeric = {'loss', 'val_loss', 'accuracy', 'val_accuracy', 'samp_s'}
    for col in cols:
        t.add_column(
            _METRIC_LABELS.get(col, col),
            justify="right" if col in numeric or col.startswith('val_') else "left",
        )
    return t


def _format_plain_metrics_table(table_state) -> str:
    cols = table_state["columns"]
    rows = table_state["rows"]
    headers = [_METRIC_LABELS.get(c, c) for c in cols]
    grid = [headers] + rows
    widths = [max(len(str(r[i])) for r in grid) for i in range(len(cols))]
    lines = []
    for ri, row in enumerate(grid):
        line = "  ".join(str(cell).rjust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(cols))))
    return "\n".join(lines)


def _dict_metrics_table(log, *, plain=None):
    use_plain = plain if plain is not None else is_plain()
    if use_plain:
        parts = [f"{k}={v if k == 'train' else _format_metric_cell(k, v)}" for k, v in log.items()]
        return "  ".join(parts)
    t = Table(box=SIMPLE_HEAVY, show_header=True, header_style="bold cyan", padding=(0, 1), expand=False)
    for k in log:
        t.add_column(k, justify="right" if k not in ('epoch', 'train') else "left")
    t.add_row(*[_styled_metric(k, v, plain=False) if k not in ('train',) else str(v) for k, v in log.items()])
    return t

def _loss_sparklines(train_losses, val_losses, max_pts=80, *, plain=None):
    spark = (
        f"train {_sparkline_str(train_losses, max_pts)}  "
        f"val {_sparkline_str(val_losses, max_pts)}"
    )
    if plain if plain is not None else is_plain():
        return spark
    return Text.assemble(
        ("train ", "dim"), (_sparkline_str(train_losses, max_pts), "cyan"),
        ("  val ", "dim"), (_sparkline_str(val_losses, max_pts), "green"),
    )
