"""Graph IR for model architecture visualization and design."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def _new_id(prefix: str = "n") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class NodeSpec:
    id: str
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    x: float = 0.0
    y: float = 0.0
    collapsed: bool = False
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["label"] is None:
            del d["label"]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeSpec:
        return cls(
            id=data["id"],
            op=data["op"],
            params=dict(data.get("params") or {}),
            x=float(data.get("x", 0)),
            y=float(data.get("y", 0)),
            collapsed=bool(data.get("collapsed", False)),
            label=data.get("label"),
        )


@dataclass
class EdgeSpec:
    id: str
    source: str
    target: str
    source_port: str = "out"
    target_port: str = "in"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EdgeSpec:
        return cls(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            source_port=data.get("source_port", "out"),
            target_port=data.get("target_port", "in"),
        )


@dataclass
class GraphSpec:
    nodes: list[NodeSpec] = field(default_factory=list)
    edges: list[EdgeSpec] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphSpec:
        return cls(
            nodes=[NodeSpec.from_dict(n) for n in data.get("nodes", [])],
            edges=[EdgeSpec.from_dict(e) for e in data.get("edges", [])],
            meta=dict(data.get("meta") or {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> GraphSpec:
        return cls.from_dict(json.loads(text))

    def node_by_id(self, node_id: str) -> NodeSpec | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def add_node(
        self,
        op: str,
        *,
        params: dict[str, Any] | None = None,
        node_id: str | None = None,
        x: float = 0,
        y: float = 0,
        label: str | None = None,
        collapsed: bool = False,
    ) -> NodeSpec:
        node = NodeSpec(
            id=node_id or _new_id("n"),
            op=op,
            params=dict(params or {}),
            x=x,
            y=y,
            label=label,
            collapsed=collapsed,
        )
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        source: str,
        target: str,
        *,
        edge_id: str | None = None,
        source_port: str = "out",
        target_port: str = "in",
    ) -> EdgeSpec:
        edge = EdgeSpec(
            id=edge_id or _new_id("e"),
            source=source,
            target=target,
            source_port=source_port,
            target_port=target_port,
        )
        self.edges.append(edge)
        return edge

    def remove_node(self, node_id: str) -> None:
        self.nodes = [n for n in self.nodes if n.id != node_id]
        self.edges = [e for e in self.edges if e.source != node_id and e.target != node_id]

    def remove_edge(self, edge_id: str) -> None:
        self.edges = [e for e in self.edges if e.id != edge_id]

    def adjacency(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            out.setdefault(edge.source, []).append(edge.target)
        return out

    def reverse_adjacency(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            out.setdefault(edge.target, []).append(edge.source)
        return out

    def topo_sort(self) -> list[str]:
        """Kahn topological sort; raises ValueError on cycle."""
        in_degree = {n.id: 0 for n in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order: list[str] = []
        adj = self.adjacency()
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for nxt in adj.get(nid, []):
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        return order

    def is_linear(self) -> bool:
        if not self.nodes:
            return True
        in_deg = {n.id: 0 for n in self.nodes}
        out_deg = {n.id: 0 for n in self.nodes}
        for edge in self.edges:
            out_deg[edge.source] = out_deg.get(edge.source, 0) + 1
            in_deg[edge.target] = in_deg.get(edge.target, 0) + 1
        return all(in_deg[n.id] <= 1 and out_deg[n.id] <= 1 for n in self.nodes)

    def auto_layout(self, *, x_gap: float = 180, y_gap: float = 90) -> None:
        """Left-to-right layout using topo order or node list order."""
        try:
            order = self.topo_sort()
        except ValueError:
            order = [n.id for n in self.nodes]
        by_id = {n.id: n for n in self.nodes}
        for i, nid in enumerate(order):
            node = by_id.get(nid)
            if node is None:
                continue
            node.x = 40 + i * x_gap
            node.y = 80 + (i % 2) * y_gap * 0.25
