"""Callback base and tensor helpers."""
from collections.abc import Mapping

import torch

class Callback(): order = 0

def run_callbacks(cbs, cb_name, learn=None):
    for cb in sorted(cbs, key=lambda x: x.order):
        if hasattr(cb, cb_name): getattr(cb, cb_name)(learn)
        
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

def to_cpu(x):
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype==torch.float16 else res

def to_device(b, device, non_blocking=False):
    if isinstance(b, tuple): return tuple(to_device(o, device, non_blocking) for o in b)
    if isinstance(b, list): return [to_device(o, device, non_blocking) for o in b]
    if isinstance(b, dict): return {k:to_device(v, device, non_blocking) for k,v in b.items()}
    return b.to(device, non_blocking=non_blocking)
