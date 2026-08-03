"""Build nn.Module from GraphSpec."""
from __future__ import annotations

import torch.nn as nn

from odyssey.viz.graph import GraphSpec
from odyssey.viz.ops import build_module, validate_graph_shapes


class GraphModule(nn.Module):
    """Execute a non-linear graph via stored topo order."""

    def __init__(self, layers: nn.ModuleDict, order: list[str], edges: list[tuple[str, str]]):
        super().__init__()
        self.layers = layers
        self.order = order
        self.edges = edges
        self._pred: dict[str, str] = {tgt: src for src, tgt in edges}

    def forward(self, x):
        cache: dict[str, object] = {}
        for nid in self.order:
            if nid not in self.layers:
                continue
        pred = self._pred.get(nid)
        if pred is None or pred not in cache:
            inp = x
        else:
            inp = cache[pred]
            out = self.layers[nid](inp)
            cache[nid] = out
        return cache[self.order[-1]] if self.order else x


def graph_to_module(graph: GraphSpec, *, validate: bool = True) -> nn.Module:
    """Compile GraphSpec into an nn.Module."""
    if validate:
        errors = validate_graph_shapes(graph)
        if errors:
            raise ValueError("Shape validation failed:\n" + "\n".join(errors))

    order = graph.topo_sort()
    buildable = [nid for nid in order if (n := graph.node_by_id(nid)) and n.op not in {"input", "group"}]

    if graph.is_linear() and len(buildable) > 0:
        modules: list[nn.Module] = []
        for nid in order:
            node = graph.node_by_id(nid)
            if node is None or node.op in {"input", "group"}:
                continue
            modules.append(build_module(node.op, node.params))
        return nn.Sequential(*modules)

    layers = nn.ModuleDict()
    edge_pairs: list[tuple[str, str]] = []
    for edge in graph.edges:
        src_node = graph.node_by_id(edge.source)
        tgt_node = graph.node_by_id(edge.target)
        if src_node and src_node.op == "input":
            continue
        if tgt_node and tgt_node.op in {"input", "group"}:
            continue
        edge_pairs.append((edge.source, edge.target))

    for nid in order:
        node = graph.node_by_id(nid)
        if node is None or node.op in {"input", "group"}:
            continue
        layers[nid] = build_module(node.op, node.params)

    return GraphModule(layers, buildable, edge_pairs)
