"""Registered layer ops for graph extract/build and shape inference."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch.nn as nn

from odyssey.models import ResBlock, Reshape, conv
from odyssey.viz.graph import GraphSpec, NodeSpec


Shape = tuple[int, ...]


@dataclass
class ParamField:
    name: str
    type: str = "int"  # int | float | bool | str
    default: Any = None
    label: str | None = None


@dataclass
class OpDef:
    name: str
    label: str
    category: str
    params: list[ParamField] = field(default_factory=list)
    buildable: bool = True
    palette: bool = True

    def default_params(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}


_OPS: dict[str, OpDef] = {}


def register_op(op: OpDef) -> OpDef:
    _OPS[op.name] = op
    return op


def get_op(name: str) -> OpDef:
    if name not in _OPS:
        raise KeyError(f"Unknown op {name!r}")
    return _OPS[name]


def list_ops(*, palette_only: bool = False) -> list[OpDef]:
    ops = list(_OPS.values())
    if palette_only:
        ops = [o for o in ops if o.palette]
    return sorted(ops, key=lambda o: (o.category, o.label))


def _ks(v: Any) -> int | tuple[int, int]:
    if isinstance(v, (list, tuple)):
        return tuple(int(x) for x in v)
    return int(v)


def _infer_conv2d(shape: Shape, params: dict[str, Any]) -> Shape:
    if len(shape) < 3:
        raise ValueError(f"conv2d expects (B,C,H,W), got {shape}")
    b = shape[0]
    _, _, h, w = shape if len(shape) == 4 else (shape[0], shape[1], 1, 1)
    out_ch = int(params.get("out_ch", params.get("out_channels", 1)))
    ks = _ks(params.get("kernel_size", 3))
    pad = params.get("padding")
    if pad is None:
        pad = ks // 2 if isinstance(ks, int) else ks[0] // 2
    stride = int(params.get("stride", 1))
    if isinstance(ks, int):
        h_out = (h + 2 * pad - ks) // stride + 1
        w_out = (w + 2 * pad - ks) // stride + 1
    else:
        h_out = (h + 2 * pad - ks[0]) // stride + 1
        w_out = (w + 2 * pad - ks[1]) // stride + 1
    return (b, out_ch, h_out, w_out)


def _build_conv2d(params: dict[str, Any]) -> nn.Module:
    in_ch = int(params.get("in_ch", params.get("in_channels", 1)))
    out_ch = int(params.get("out_ch", params.get("out_channels", 1)))
    ks = _ks(params.get("kernel_size", 3))
    stride = int(params.get("stride", 1))
    pad = params.get("padding")
    if pad is None:
        pad = ks // 2 if isinstance(ks, int) else ks[0] // 2
    act = bool(params.get("act", True))
    if act:
        return conv(in_ch, out_ch, ks=ks if isinstance(ks, int) else ks[0], act=True)
    return nn.Conv2d(in_ch, out_ch, ks, stride=stride, padding=pad, bias=True)


def _infer_reshape(shape: Shape, params: dict[str, Any]) -> Shape:
    b = shape[0]
    dims = params.get("shape") or params.get("dims")
    if dims is None:
        raise ValueError("reshape requires shape/dims params")
    tail = tuple(int(x) for x in dims)
    return (b, *tail)


def _build_reshape(params: dict[str, Any]) -> nn.Module:
    dims = params.get("shape") or params.get("dims")
    if dims is None:
        raise ValueError("reshape requires shape/dims params")
    return Reshape(*tuple(int(x) for x in dims))


def _infer_resblock(shape: Shape, params: dict[str, Any]) -> Shape:
    if len(shape) != 4:
        raise ValueError(f"resblock expects (B,C,H,W), got {shape}")
    b, _, h, w = shape
    out_ch = int(params.get("out_ch", params.get("nf", shape[1])))
    return (b, out_ch, h, w)


def _build_resblock(params: dict[str, Any]) -> nn.Module:
    in_ch = int(params.get("in_ch", params.get("ni", 1)))
    out_ch = int(params.get("out_ch", params.get("nf", in_ch)))
    return ResBlock(in_ch, out_ch)


def _infer_linear(shape: Shape, params: dict[str, Any]) -> Shape:
    b = shape[0]
    in_f = int(params.get("in_features", params.get("n_in", 784)))
    flat = math.prod(shape[1:]) if len(shape) > 1 else shape[-1] if shape else 0
    if len(shape) > 1 and flat != in_f:
        raise ValueError(f"linear expects {in_f} features, got {flat} from shape {shape}")
    out_f = int(params.get("out_features", params.get("n_out", 10)))
    return (b, out_f)


def _build_linear(params: dict[str, Any]) -> nn.Module:
    in_f = int(params.get("in_features", params.get("n_in", 784)))
    out_f = int(params.get("out_features", params.get("n_out", 10)))
    return nn.Linear(in_f, out_f)


def _infer_flatten(shape: Shape, params: dict[str, Any]) -> Shape:
    b = shape[0]
    flat = math.prod(shape[1:]) if len(shape) > 1 else 1
    return (b, flat)


def _infer_adaptive_avg_pool2d(shape: Shape, params: dict[str, Any]) -> Shape:
    if len(shape) != 4:
        raise ValueError(f"adaptive_avg_pool2d expects (B,C,H,W), got {shape}")
    b, c = shape[0], shape[1]
    out = int(params.get("output_size", 1))
    return (b, c, out, out)


def _infer_batchnorm2d(shape: Shape, params: dict[str, Any]) -> Shape:
    if len(shape) != 4:
        raise ValueError(f"batchnorm2d expects (B,C,H,W), got {shape}")
    return shape


def _infer_activation(shape: Shape, params: dict[str, Any]) -> Shape:
    return shape


def _infer_input(shape: Shape, params: dict[str, Any]) -> Shape:
    dims = params.get("shape") or params.get("dims")
    if dims:
        return (1, *tuple(int(x) for x in dims))
    return shape


def _build_activation(kind: str) -> Callable[[dict[str, Any]], nn.Module]:
    def _factory(params: dict[str, Any]) -> nn.Module:
        if kind == "relu":
            return nn.ReLU(inplace=True)
        if kind == "silu":
            return nn.SiLU(inplace=True)
        raise ValueError(kind)

    return _factory


def _build_group(params: dict[str, Any]) -> nn.Module:
    raise ValueError("group nodes are view-only and cannot be built")


_INFER: dict[str, Callable[[Shape, dict[str, Any]], Shape]] = {
    "input": _infer_input,
    "reshape": _infer_reshape,
    "conv2d": _infer_conv2d,
    "conv": _infer_conv2d,
    "resblock": _infer_resblock,
    "linear": _infer_linear,
    "flatten": _infer_flatten,
    "adaptive_avg_pool2d": _infer_adaptive_avg_pool2d,
    "batchnorm2d": _infer_batchnorm2d,
    "relu": _infer_activation,
    "silu": _infer_activation,
    "group": lambda s, p: s,
}

_BUILD: dict[str, Callable[[dict[str, Any]], nn.Module]] = {
    "reshape": _build_reshape,
    "conv2d": _build_conv2d,
    "conv": _build_conv2d,
    "resblock": _build_resblock,
    "linear": _build_linear,
    "flatten": lambda p: nn.Flatten(),
    "adaptive_avg_pool2d": lambda p: nn.AdaptiveAvgPool2d(int(p.get("output_size", 1))),
    "batchnorm2d": lambda p: nn.BatchNorm2d(int(p.get("num_features", p.get("out_ch", 1)))),
    "relu": _build_activation("relu"),
    "silu": _build_activation("silu"),
    "group": _build_group,
}


def infer_shape(op: str, in_shape: Shape, params: dict[str, Any]) -> Shape:
    fn = _INFER.get(op)
    if fn is None:
        return in_shape
    return fn(in_shape, params)


def build_module(op: str, params: dict[str, Any]) -> nn.Module:
    if op == "input":
        raise ValueError("input nodes do not produce modules")
    fn = _BUILD.get(op)
    if fn is None:
        raise ValueError(f"Cannot build op {op!r}")
    return fn(params)


def validate_graph_shapes(graph: GraphSpec) -> list[str]:
    """Return list of shape error messages (empty if valid)."""
    errors: list[str] = []
    input_shape = tuple(graph.meta.get("input_shape") or (1, 1, 28, 28))
    shapes: dict[str, Shape] = {}
    rev = graph.reverse_adjacency()
    try:
        order = graph.topo_sort()
    except ValueError:
        return ["Graph contains a cycle"]

    for nid in order:
        node = graph.node_by_id(nid)
        if node is None:
            continue
        preds = rev.get(nid, [])
        if not preds:
            if node.op == "input":
                shapes[nid] = infer_shape("input", input_shape, node.params)
            else:
                shapes[nid] = infer_shape(node.op, input_shape, node.params)
        else:
            pred = preds[0]
            in_shape = shapes.get(pred)
            if in_shape is None:
                errors.append(f"Node {node.id}: missing input shape from {pred}")
                continue
            try:
                shapes[nid] = infer_shape(node.op, in_shape, node.params)
            except ValueError as exc:
                errors.append(f"Node {node.label or node.op} ({node.id}): {exc}")
    return errors


def params_from_module(mod: nn.Module) -> dict[str, Any]:
    """Snapshot params from a known nn.Module instance."""
    if isinstance(mod, Reshape):
        return {"shape": list(mod.shape)}
    if isinstance(mod, ResBlock):
        conv1 = mod.conv1[0]
        return {"in_ch": conv1.in_channels, "out_ch": conv1.out_channels}
    if isinstance(mod, nn.Conv2d):
        ks = mod.kernel_size[0] if isinstance(mod.kernel_size, tuple) else mod.kernel_size
        return {
            "in_ch": mod.in_channels,
            "out_ch": mod.out_channels,
            "kernel_size": ks,
            "stride": mod.stride[0] if isinstance(mod.stride, tuple) else mod.stride,
            "padding": mod.padding[0] if isinstance(mod.padding, tuple) else mod.padding,
        }
    if isinstance(mod, nn.Linear):
        return {"in_features": mod.in_features, "out_features": mod.out_features}
    if isinstance(mod, nn.BatchNorm2d):
        return {"num_features": mod.num_features, "out_ch": mod.num_features}
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        out = mod.output_size
        if isinstance(out, tuple):
            out = out[0]
        return {"output_size": out}
    if isinstance(mod, nn.Sequential) and len(mod) == 2:
        if isinstance(mod[0], nn.AdaptiveAvgPool2d) and isinstance(mod[1], (nn.Flatten, nn.Linear)):
            if isinstance(mod[1], nn.Linear):
                return {
                    "output_size": mod[0].output_size[0]
                    if isinstance(mod[0].output_size, tuple)
                    else mod[0].output_size,
                    "in_features": mod[1].in_features,
                    "out_features": mod[1].out_features,
                    "op": "head",
                }
    return {"label": type(mod).__name__}


def op_name_for_module(mod: nn.Module) -> str | None:
    if isinstance(mod, Reshape):
        return "reshape"
    if isinstance(mod, ResBlock):
        return "resblock"
    if isinstance(mod, nn.Conv2d):
        return "conv2d"
    if isinstance(mod, nn.Linear):
        return "linear"
    if isinstance(mod, nn.ReLU):
        return "relu"
    if isinstance(mod, nn.SiLU):
        return "silu"
    if isinstance(mod, nn.Flatten):
        return "flatten"
    if isinstance(mod, nn.BatchNorm2d):
        return "batchnorm2d"
    if isinstance(mod, nn.AdaptiveAvgPool2d):
        return "adaptive_avg_pool2d"
    if isinstance(mod, nn.Identity):
        return None
    if isinstance(mod, nn.Sequential):
        if len(mod) >= 2 and isinstance(mod[0], nn.AdaptiveAvgPool2d):
            return "head"
        if len(mod) == 2 and isinstance(mod[0], nn.Conv2d) and isinstance(mod[1], (nn.SiLU, nn.ReLU)):
            return "conv"
    return None


# Register palette ops
register_op(OpDef("input", "Input", "io", [
    ParamField("shape", "str", "[1,28,28]", "Shape (no batch)"),
], buildable=False))
register_op(OpDef("reshape", "Reshape", "shape", [
    ParamField("shape", "str", "[1,28,28]", "Shape (no batch)"),
]))
register_op(OpDef("conv2d", "Conv2d", "conv", [
    ParamField("in_ch", "int", 1),
    ParamField("out_ch", "int", 8),
    ParamField("kernel_size", "int", 3),
    ParamField("stride", "int", 1),
    ParamField("padding", "int", 1),
    ParamField("act", "bool", False),
]))
register_op(OpDef("conv", "Conv+SiLU", "conv", [
    ParamField("in_ch", "int", 1),
    ParamField("out_ch", "int", 8),
    ParamField("kernel_size", "int", 3),
]))
register_op(OpDef("resblock", "ResBlock", "conv", [
    ParamField("in_ch", "int", 8),
    ParamField("out_ch", "int", 16),
]))
register_op(OpDef("linear", "Linear", "dense", [
    ParamField("in_features", "int", 784),
    ParamField("out_features", "int", 10),
]))
register_op(OpDef("flatten", "Flatten", "shape", []))
register_op(OpDef("adaptive_avg_pool2d", "AdaptiveAvgPool2d", "pool", [
    ParamField("output_size", "int", 1),
]))
register_op(OpDef("batchnorm2d", "BatchNorm2d", "norm", [
    ParamField("num_features", "int", 8),
]))
register_op(OpDef("relu", "ReLU", "act", []))
register_op(OpDef("silu", "SiLU", "act", []))
register_op(OpDef("group", "Group", "meta", [
    ParamField("label", "str", "Submodule"),
], buildable=False, palette=False))
