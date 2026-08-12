"""Model diagnostics and activation histogram plots."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from fastai.callback.hook import flatten_model
from fastcore.basics import PrettyString
from rich.box import SIMPLE_HEAVY
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from odyssey.models import ResBlock, Reshape
from odyssey.tracking.ui import (
    console,
    describe_group_depth,
    describe_max_layers,
    is_plain,
    safe_plt_show,
    top_params,
)

_TOP_PARAMS = top_params()
_DESCRIBE_MAX_LAYERS = describe_max_layers()
_DESCRIBE_GROUP_DEPTH = describe_group_depth()

def _activation_layer_label(name, mod):
    "Human-readable label for a main activation hook target."
    if isinstance(mod, ResBlock):
        return f"ResBlock {name}" if name.isdigit() else f"ResBlock {name.replace('.', '/')}"
    if isinstance(mod, nn.Conv2d):
        return f"Conv2d {name.replace('.', '/')}"
    if isinstance(mod, nn.Linear):
        return f"Linear {name.replace('.', '/')}"
    return name.replace('.', '/')

def _activation_layer_sort_key(label):
    parts = label.split()
    if len(parts) >= 2 and parts[-1].isdigit(): return (0, int(parts[-1]))
    return (1, label)

def _activation_hook_layers(model):
    "Activation hook targets across common architectures (CNN/ResNet/MLP)."
    # Avoid double-counting convs inside residual blocks: hook the ResBlock itself, not its internals.
    resblock_internals = set()
    for rb in model.modules():
        if isinstance(rb, ResBlock):
            for sm in rb.modules():
                if sm is rb:
                    continue
                resblock_internals.add(id(sm))
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, ResBlock):
            layers.append((_activation_layer_label(name, mod), mod))
        elif isinstance(mod, nn.Conv2d) and id(mod) not in resblock_internals:
            layers.append((_activation_layer_label(name, mod), mod))
        elif isinstance(mod, nn.Linear):
            layers.append((_activation_layer_label(name, mod), mod))
    layers.sort(key=lambda x: _activation_layer_sort_key(x[0]))
    return layers

def _count_dead_neurons(tensors, eps=1e-6):
    "Channels with per-channel std (over batch+space) below eps."
    if not tensors: return None
    t = torch.cat(tensors, dim=0)
    if t.dim() < 2: return {'dead': 0, 'n_neurons': 1}
    reduce_dims = (0,) + tuple(range(2, t.dim()))
    per_ch_std = t.std(dim=reduce_dims, unbiased=False)
    dead = int((per_ch_std < eps).sum())
    return {'dead': dead, 'n_neurons': int(per_ch_std.numel())}

def _activation_log1p_abs_stats(tensors, n_bins, *, dead_eps=1e-6):
    "Mean, std, histc of log1p(|activation|), and dead-neuron count (no subsampling)."
    if not tensors: return None
    dead_info = _count_dead_neurons(tensors, dead_eps)
    if dead_info is None: return None
    t = torch.cat([x.flatten() for x in tensors])
    t = t[torch.isfinite(t)]
    if t.numel() == 0: return None
    v = torch.log1p(t.abs())
    mean = float(v.mean())
    std = float(v.std(unbiased=False)) if v.numel() > 1 else 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        hi = lo + 1e-6
    hist = torch.histc(v, bins=n_bins, min=lo, max=hi).float().cpu().numpy()
    return {'mean': mean, 'std': std, 'hist': hist, 'lo': lo, 'hi': hi, **dead_info}

def _rebin_hist(counts, src_edges, dst_edges):
    "Redistribute histogram counts from src_edges to dst_edges by bin overlap."
    out = np.zeros(len(dst_edges) - 1, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.float64)
    for i, c in enumerate(counts):
        if c == 0: continue
        lo, hi = src_edges[i], src_edges[i + 1]
        width = hi - lo
        if width <= 0:
            j = int(np.clip(np.searchsorted(dst_edges, lo) - 1, 0, len(out) - 1))
            out[j] += c
            continue
        j0 = max(0, int(np.searchsorted(dst_edges, lo, side='right') - 1))
        j1 = min(len(out), int(np.searchsorted(dst_edges, hi, side='left')) + 1)
        for j in range(j0, j1):
            overlap = max(0.0, min(hi, dst_edges[j + 1]) - max(lo, dst_edges[j]))
            out[j] += c * overlap / width
    return out

def _activation_bin_matrix(entries, n_epochs, epoch_idx, n_bins=96):
    "Build (n_bins, n_epochs) count matrix for one layer from log1p(|a|) stats."
    global_lo = min(s['lo'] for _, s in entries)
    global_hi = max(s['hi'] for _, s in entries)
    if global_hi <= global_lo:
        global_hi = global_lo + 1e-6
    edges = np.linspace(global_lo, global_hi, n_bins + 1)
    mat = np.zeros((n_bins, n_epochs), dtype=np.float32)
    for epoch, s in entries:
        col = epoch_idx.get(epoch)
        if col is None: continue
        src_edges = np.linspace(s['lo'], s['hi'], len(s['hist']) + 1)
        mat[:, col] = _rebin_hist(s['hist'], src_edges, edges)
    return mat, float(edges[0]), float(edges[-1]), edges

def plot_activation_hists(history, *, n_bins=96, save_path=None, show=False,
                          title="log1p(|activation|) heatmap (epoch × bin)"):
    "Per-layer heatmaps: x=epoch, y=log1p(|a|) bin, pixel=bin count; title shows latest μ±σ."
    layers = sorted((k for k, v in history.items() if v), key=_activation_layer_sort_key)
    if not layers: return None
    global_epochs = sorted({e for vals in history.values() for e, _ in vals})
    n_epochs = len(global_epochs)
    epoch_idx = {e: i for i, e in enumerate(global_epochs)}

    cols = 2 if len(layers) > 1 else 1
    rows = math.ceil(len(layers) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.4), squeeze=False)
    for i, layer in enumerate(layers):
        ax = axes[i // cols][i % cols]
        mat, lo, hi, edges = _activation_bin_matrix(history[layer], n_epochs, epoch_idx, n_bins=n_bins)
        vmax = mat.max() or 1.0
        ax.imshow(mat, aspect='auto', origin='lower', cmap='viridis',
                  interpolation='nearest', vmin=0, vmax=vmax)
        last = history[layer][-1][1]
        ax.set_title(
            f"{layer}\nμ={last['mean']:.3g} σ={last['std']:.3g} "
            f"dead={last['dead']}/{last['n_neurons']}",
            fontsize=7,
        )
        ax.set_xlabel('epoch', fontsize=7)
        ax.set_ylabel('log1p(|a|)', fontsize=7)
        if n_epochs <= 16:
            ax.set_xticks(range(n_epochs))
            ax.set_xticklabels([str(e) for e in global_epochs], fontsize=6)
        else:
            ax.set_xticks([])
        mid = len(edges) // 2
        ax.set_yticks([0, mid, len(edges) - 2])
        ax.set_yticklabels([f'{lo:.2g}', f'{edges[mid]:.2g}', f'{hi:.2g}'], fontsize=5)
    for j in range(len(layers), rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    if title: fig.suptitle(title, fontsize=9, y=1.02)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        # _console.print(f"[green]Activation heatmap saved → {save_path}[/green]")
    _safe_plt_show(fig, show)
    return fig

def plot_dead_neurons(history, *, save_path=None, show=False, title="Dead channels (%) per layer × epoch"):
    "Line plot of dead-channel percentage per layer across epochs."
    layers = sorted((k for k, v in history.items() if v), key=_activation_layer_sort_key)
    if not layers: return None
    fig, ax = plt.subplots(figsize=(max(4, len(layers) * 1.2), 3))
    for layer in layers:
        epochs = [e for e, _ in history[layer]]
        dead_pcts = [
            (100.0 * s['dead'] / s['n_neurons']) if s.get('n_neurons') else 0.0
            for _, s in history[layer]
        ]
        ax.plot(epochs, dead_pcts, marker='o', ms=3, label=layer)
    ax.set_xlabel('epoch', fontsize=8)
    ax.set_ylabel('dead channels (%)', fontsize=8)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.3)
    if title: ax.set_title(title, fontsize=9)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    _safe_plt_show(fig, show)
    return fig

def _entry_inp_size(m):
    "Input shape at `m.forward` entry (e.g. flat vector if model starts with `Reshape`)."
    for mod in flatten_model(m):
        if isinstance(mod, Reshape): return (mod.flat_dim,)
        if isinstance(mod, nn.Linear): return (mod.in_features,)
        if isinstance(mod, nn.Conv2d): return (mod.in_channels, 28, 28)
    return None

def _example_x(m, inp_size, ex_inp, bs, device):
    if ex_inp is not None: return ex_inp.to(device)
    inp_size = inp_size or _entry_inp_size(m)
    if inp_size is None: return None
    return torch.zeros((bs, *inp_size), device=device)

def _thop_macs(m, x):
    try:
        from copy import deepcopy
        from thop import profile
        was = m.training
        probe = deepcopy(m)
        probe.eval()
        with torch.no_grad(): macs,_ = profile(probe, inputs=(x,), verbose=False)
        m.train(was)
        return macs, None
    except Exception as e: return None, str(e)

def _mem(n):
    if n>=(1<<30): return f'{n/(1<<30):.2f} GiB'
    if n>=(1<<20): return f'{n/(1<<20):.2f} MiB'
    if n>=(1<<10): return f'{n/(1<<10):.2f} KiB'
    return f'{n} B'

def _ops_fmt(n, unit):
    "Format op count as K/M/G `{unit}`s (e.g. MFLOPs, GMACs)."
    if n >= 1e9: return f'{n/1e9:.2f} G{unit}s'
    if n >= 1e6: return f'{n/1e6:.2f} M{unit}s'
    if n >= 1e3: return f'{n/1e3:.2f} K{unit}s'
    return f'{int(n)} {unit}s'

def _pct(n, total): return f'{100*n/total:5.1f}%' if total else '  n/a '

def _group_name(path: str, depth: int) -> str:
    if not path:
        return "model"
    parts = path.split(".")
    return ".".join(parts[: min(depth, len(parts))])

def _aggregate_layer_stats(m, mod_macs, group_depth: int = _DESCRIBE_GROUP_DEPTH):
    "Sum params and MACs into named module groups (e.g. resnet blocks at depth 2)."
    groups = {}
    for name, param in m.named_parameters():
        mod_path = name.rsplit(".", 1)[0] if "." in name else ""
        gname = _group_name(mod_path, group_depth)
        g = groups.setdefault(gname, {"params": 0, "macs": 0, "trainable": False})
        g["params"] += param.numel()
        g["trainable"] = g["trainable"] or param.requires_grad
    name_map = {id(mod): (name or "model") for name, mod in m.named_modules()}
    for mod, macs in mod_macs.items():
        full = name_map.get(id(mod), "model")
        gname = _group_name(full, group_depth)
        groups.setdefault(gname, {"params": 0, "macs": 0, "trainable": False})["macs"] += macs
    return groups

def _collapse_layer_stats(groups, max_rows: int = _DESCRIBE_MAX_LAYERS):
    "Keep the table readable for deep nets: top layers by FLOPs + rolled-up remainder."
    items = sorted(groups.items(), key=lambda kv: kv[1]["macs"], reverse=True)
    if len(items) <= max_rows:
        return items
    head = items[: max_rows - 1]
    tail = items[max_rows - 1 :]
    others = {"params": 0, "macs": 0, "trainable": False}
    for _, st in tail:
        others["params"] += st["params"]
        others["macs"] += st["macs"]
        others["trainable"] = others["trainable"] or st["trainable"]
    return head + [(f"others (+{len(tail)} layers)", others)]

def _pick_layer_rows(m, mod_macs, max_rows: int = _DESCRIBE_MAX_LAYERS):
    "Prefer coarse groups when depth-2 still yields too many rows before collapse."
    for depth in (_DESCRIBE_GROUP_DEPTH, 1, 3):
        groups = _aggregate_layer_stats(m, mod_macs, depth)
        rows = _collapse_layer_stats(groups, max_rows)
        if len(groups) <= max_rows or depth == 1:
            return rows, depth
    return _collapse_layer_stats(_aggregate_layer_stats(m, mod_macs, 1), max_rows), 1

def _macs_by_module(m, x):
    "Estimated forward MACs per module (Linear / Conv2d) on one forward pass."
    stats = {}
    def hook(mod, inp, out):
        inp_t = inp[0] if isinstance(inp, tuple) else inp
        if isinstance(mod, nn.Linear):
            b = inp_t.shape[0]
            stats[mod] = b * mod.in_features * mod.out_features
        elif isinstance(mod, nn.Conv2d):
            o = out if isinstance(out, torch.Tensor) else out[0]
            ks = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size)
            stats[mod] = o.numel() * math.prod(ks) * mod.in_channels // mod.groups
    hs = [o.register_forward_hook(hook) for o in m.modules() if isinstance(o, (nn.Linear, nn.Conv2d))]
    was = m.training
    m.eval()
    with torch.no_grad(): m(x)
    m.train(was)
    for h in hs: h.remove()
    return stats

def describe_model(m, inp_size=None, ex_inp=None, bs=1, device=None, title=None):
    "Summary of `m`: params, memory, layers, and optional MAC/FLOP estimate (thop)."
    p = next(m.parameters(), None)
    if device is None:
        device = str(p.device) if p is not None else "cpu"
    m = m.to(device)
    dev,dtype = (str(p.device), str(p.dtype).replace('torch.','')) if p is not None else (device, 'float32')
    inp_size = inp_size or _entry_inp_size(m)
    x = _example_x(m, inp_size, ex_inp, bs, p.device if p is not None else device)

    bufs = sum(b.numel() for b in m.buffers())
    pmem = sum(o.numel()*o.element_size() for o in m.parameters())
    bmem = sum(o.numel()*o.element_size() for o in m.buffers())
    mod_macs = _macs_by_module(m, x) if x is not None else {}
    total_macs = sum(mod_macs.values()) or 0
    total_flops = 2 * total_macs

    ps, trn_ps = 0, 0
    for _, param in m.named_parameters():
        ps += param.numel()
        if param.requires_grad:
            trn_ps += param.numel()

    macs, err = _thop_macs(m, x) if x is not None else (None, None)
    c = console()
    plain = is_plain()

    with c.capture() as cap:
        hdr = title or type(m).__name__
        if plain:
            meta_bits = [f"params {ps:,}", f"mem {_mem(pmem + bmem)}"]
            if x is not None:
                meta_bits.append(f"in {tuple(x.shape)}")
            meta_bits.append(f"{dev} {dtype}")
            c.print(f"=== {hdr} ===")
            c.print(" · ".join(meta_bits))
        else:
            meta = Text.assemble(
                ("params ", "dim"), (f"{ps:,}", "bold cyan"), (" · mem ", "dim"), (_mem(pmem + bmem), "bold yellow"),
            )
            if x is not None:
                meta.append_text(Text.assemble((" · in ", "dim"), (str(tuple(x.shape)), "white")))
            meta.append_text(Text.assemble((" · ", "dim"), (dev, "dim"), (" ", ""), (dtype, "dim")))
            c.print(Panel(meta, title=f"[bold cyan]{hdr}[/bold cyan]", border_style="cyan", padding=(0, 1)))

        if mod_macs:
            layer_rows, group_depth = _pick_layer_rows(m, mod_macs)
            if plain:
                c.print(f"FLOPs by layer (group depth {group_depth})")
                c.print(f"{'Layer':<28} {'Params':>12} {'FLOPs':>12} {'%FLOPs':>8} {'%Params':>8} Tr")
                for gname, st in layer_rows:
                    flops = 2 * st["macs"]
                    pct_f = 100 * st["macs"] / total_macs if total_macs else 0
                    pct_p = 100 * st["params"] / ps if ps else 0
                    tr = "Y" if st["trainable"] else "-"
                    c.print(
                        f"{gname:<28} {st['params']:>12,} {_ops_fmt(flops, 'FLOP'):>12} "
                        f"{pct_f:7.1f}% {pct_p:7.1f}% {tr}"
                    )
            else:
                lt = Table(
                    title=f"FLOPs by layer (group depth {group_depth})",
                    box=SIMPLE_HEAVY,
                    header_style="bold cyan",
                    show_lines=False,
                    padding=(0, 1),
                    show_edge=False,
                    expand=False,
                )
                lt.add_column("Layer", style="bold", no_wrap=False, max_width=28)
                lt.add_column("Params", justify="right")
                lt.add_column("FLOPs", justify="right", style="magenta")
                lt.add_column("%FLOPs", justify="right", style="dim")
                lt.add_column("%Params", justify="right", style="dim")
                lt.add_column("Tr", justify="center")
                for gname, st in layer_rows:
                    flops = 2 * st["macs"]
                    pct_f = 100 * st["macs"] / total_macs if total_macs else 0
                    pct_p = 100 * st["params"] / ps if ps else 0
                    row_style = "bold yellow" if pct_f >= 15 else None
                    lt.add_row(
                        Text(gname, style=row_style),
                        Text(f"{st['params']:,}", style=row_style),
                        Text(_ops_fmt(flops, "FLOP"), style=row_style or "magenta"),
                        _pct(st["macs"], total_macs),
                        _pct(st["params"], ps),
                        Text("Y", style="green") if st["trainable"] else Text("-", style="dim"),
                    )
                c.print(lt)

        if plain:
            stats = f"trainable {trn_ps:,} / {ps:,} · buf {bufs:,} ({_mem(bmem)})"
            if err:
                stats += " · profile skipped"
            elif macs is not None:
                stats += f" · MACs {_ops_fmt(macs, 'MAC')} · FLOPs {_ops_fmt(2 * macs, 'FLOP')}"
            elif total_macs:
                stats += f" · MACs {_ops_fmt(total_macs, 'MAC')} · FLOPs {_ops_fmt(total_flops, 'FLOP')}"
            c.print(stats)
        else:
            stats = Text.assemble(
                ("trainable ", "dim"), (f"{trn_ps:,}", "green"), (" / ", "dim"), (f"{ps:,}", "cyan"),
                (" · buf ", "dim"), (f"{bufs:,}", "dim"), (f" ({_mem(bmem)})", "dim"),
            )
            if err:
                stats.append_text(Text.assemble((" · ", "dim"), ("profile skipped", "dim red")))
            elif macs is not None:
                stats.append_text(Text.assemble(
                    (" · MACs ", "dim"), (_ops_fmt(macs, "MAC"), "bold magenta"),
                    (" · FLOPs ", "dim"), (_ops_fmt(2 * macs, "FLOP"), "magenta"),
                ))
            elif total_macs:
                stats.append_text(Text.assemble(
                    (" · MACs ", "dim"), (_ops_fmt(total_macs, "MAC"), "bold magenta"),
                    (" · FLOPs ", "dim"), (_ops_fmt(total_flops, "FLOP"), "magenta"),
                ))
            c.print(stats)

        tensors = sorted(m.named_parameters(), key=lambda t: t[1].numel(), reverse=True)[:_TOP_PARAMS]
        n_all = sum(1 for _ in m.named_parameters())
        if len(tensors) < n_all:
            n_more = n_all - len(tensors)
            pt_title = f"Top {_TOP_PARAMS} params (+{n_more} more)"
        else:
            pt_title = "Params"
        if plain:
            c.print(pt_title)
            for nm, o in tensors:
                n = o.numel()
                c.print(f"  {nm}  {tuple(o.shape)}  {n:,}  {_mem(n * o.element_size())}")
        else:
            pt = Table(title=pt_title, box=SIMPLE_HEAVY, header_style="bold cyan",
                       padding=(0, 1), show_edge=False, expand=False)
            pt.add_column("Name")
            pt.add_column("Shape", style="dim")
            pt.add_column("#", justify="right")
            pt.add_column("Mem", justify="right", style="yellow")
            for nm,o in tensors:
                n = o.numel()
                pt.add_row(nm, str(tuple(o.shape)), f"{n:,}", _mem(n*o.element_size()))
            c.print(pt)

    return PrettyString(cap.get())
