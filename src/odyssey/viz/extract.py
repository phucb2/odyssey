"""Extract GraphSpec from nn.Module."""
from __future__ import annotations

import torch.nn as nn

from odyssey.viz.graph import GraphSpec, _new_id
from odyssey.viz.ops import op_name_for_module, params_from_module


def _infer_input_shape(module: nn.Module) -> tuple[int, ...]:
    for mod in module.modules():
        if hasattr(mod, "shape"):
            shape = getattr(mod, "shape")
            if isinstance(shape, tuple):
                return (1, *shape)
        if isinstance(mod, nn.Conv2d):
            return (1, mod.in_channels, 28, 28)
        if isinstance(mod, nn.Linear):
            return (1, mod.in_features)
    return (1, 1, 28, 28)


def _expand_module(name: str, mod: nn.Module) -> list[tuple[str, nn.Module]]:
    op = op_name_for_module(mod)
    if op == "head":
        return [(f"{name}.{i}", sub) for i, sub in enumerate(mod)]
    if op == "conv":
        return [(name, mod)]
    if op is not None:
        return [(name, mod)]
    if list(mod.children()):
        return [(name, mod)]
    return [(name, mod)]


def _add_module_node(graph: GraphSpec, name: str, mod: nn.Module) -> str | None:
    op = op_name_for_module(mod)
    if op == "conv":
        nid = _new_id("n")
        graph.add_node(op, params=params_from_module(mod[0]), node_id=nid, label=name)
        return nid
    if op is None and list(mod.children()):
        nid = _new_id("n")
        graph.add_node(
            "group",
            params={"label": name or type(mod).__name__},
            node_id=nid,
            label=name or type(mod).__name__,
            collapsed=True,
        )
        return nid
    if op is None:
        return None
    nid = _new_id("n")
    graph.add_node(op, params=params_from_module(mod), node_id=nid, label=name)
    return nid


def _extract_detr(module: nn.Module) -> GraphSpec:
    graph = GraphSpec(meta={"title": type(module).__name__, "input_shape": (1, 3, 320, 320)})
    prev: str | None = None
    for name, child in module.named_children():
        op = op_name_for_module(child)
        if op is None:
            nid = _new_id("n")
            graph.add_node(
                "group",
                params={"label": name},
                node_id=nid,
                label=name,
                collapsed=True,
            )
        else:
            nid = _new_id("n")
            graph.add_node(op, params=params_from_module(child), node_id=nid, label=name)
        if prev:
            graph.add_edge(prev, nid)
        prev = nid
    graph.auto_layout()
    return graph


def module_to_graph(
    module: nn.Module,
    *,
    title: str | None = None,
    input_shape: tuple[int, ...] | None = None,
) -> GraphSpec:
    """Convert an nn.Module into a GraphSpec for visualization/editing."""
    if type(module).__name__ == "SmallDETR":
        return _extract_detr(module)

    in_shape = input_shape or _infer_input_shape(module)
    graph = GraphSpec(meta={"title": title or type(module).__name__, "input_shape": in_shape})

    if isinstance(module, nn.Sequential):
        raw = [(str(i), m) for i, m in enumerate(module)]
    else:
        raw = list(module.named_children())

    modules: list[tuple[str, nn.Module]] = []
    for name, mod in raw:
        modules.extend(_expand_module(name, mod))

    input_id = graph.add_node("input", params={"shape": list(in_shape[1:])}).id
    prev = input_id
    for name, mod in modules:
        nid = _add_module_node(graph, name, mod)
        if nid:
            graph.add_edge(prev, nid)
            prev = nid

    graph.auto_layout()
    return graph
