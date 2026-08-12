# Plain console mode (`ODYSSEY_PLAIN`)

Used when training progress / metrics should print as plain text instead of Rich Live panels. Set before training (CLI env or `set_plain`) when logs go to files, CI, or you prefer text-only output.

## What it does

| Mode | Behavior |
|------|----------|
| **Rich** (default) | Live progress bars, colored tables, panels |
| **Plain** | Throttled status lines + plain metric / describe text; no Live |

Covers training progress (`ProgressCB`), metrics tables, and `describe_model`. Matplotlib / TensorBoard are unchanged.

## Enable

### Environment

```bash
ODYSSEY_PLAIN=1 uv run mltrain ...
```

Accepted values: `1`, `true`, `yes`, `on` (case-insensitive).

### Python API

```python
from odyssey.tracking import set_plain, is_plain

set_plain(True)   # force plain
set_plain(False)  # force Rich
set_plain(None)   # follow ODYSSEY_PLAIN again

assert is_plain() is True
```

### One-run callback override

```python
from odyssey.training import ProgressCB

cbs = [..., ProgressCB(plot=True, plain=True)]
```

`plain=None` (default) follows `is_plain()`; `True` / `False` overrides for that callback only.

## Example plain output

```
epochs 0/10 | train loss 0.321 · 12.3ms · 820 s/s 45/100
epoch 1/10 · loss 0.3200 · acc 0.9100 · vloss 0.2800*
 epoch  loss   acc  vloss  ...
 -----  ----  ----  -----
     1  0.32  0.91  0.28
```

Mid-epoch refreshes print only the status line; epoch / metric logs print the summary (+ optional history table and loss sparklines when `plot=True`).

## Notes

- Set `ODYSSEY_PLAIN` (or call `set_plain`) **before** training starts so the console is rebuilt correctly.
- `ProgressCB(plain=True)` does not change global `is_plain()`; other Rich printers (e.g. LR finder panels) still follow the global flag / env.
