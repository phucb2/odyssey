"""Weight initialization presets."""
import torch.nn as nn

from odyssey.training.callback import Callback

def init_kaiming(m, nonlinearity='relu'):
    "Kaiming normal for Linear / Conv2d; zeros bias."
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None: nn.init.zeros_(m.bias)

def init_xavier(m):
    "Xavier uniform for Linear / Conv2d; zeros bias."
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None: nn.init.zeros_(m.bias)

def init_trunc_normal(m, std=0.02):
    "Truncated normal for Linear / Conv2d; LayerNorm to 1/0."
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None: nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

_INIT_PRESETS = {'kaiming': init_kaiming, 'xavier': init_xavier, 'trunc_normal': init_trunc_normal}

def resolve_init(init):
    "Turn preset name or callable into a module-wise init function."
    if init is None: return None
    if isinstance(init, str):
        if init not in _INIT_PRESETS:
            raise ValueError(f"Unknown init preset {init!r}; choose from {tuple(_INIT_PRESETS)}")
        return _INIT_PRESETS[init]
    if not callable(init): raise TypeError(f"init must be a preset name or callable, got {type(init)}")
    return init

def apply_init(model, init):
    "Apply `init` to `model` immediately (outside the callback pipeline)."
    fn = resolve_init(init)
    if fn is not None: model.apply(fn)

class InitCB(Callback):
    "Apply weight init at `before_fit`; omit from `cbs` or pass `init=None` to skip."
    order = -5
    def __init__(self, init):
        self.init = resolve_init(init)
    def before_fit(self, learn):
        if self.init is not None: learn.model.apply(self.init)

def _merge_init_cbs(cbs, init):
    "Prepend InitCB when `init` is set; leave `cbs` unchanged when `init` is None."
    cbs = [] if cbs is None else list(cbs)
    if init is not None: cbs.insert(0, InitCB(init))
    return cbs
