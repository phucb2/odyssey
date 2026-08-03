"""Interactive architecture editor anywidget."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import anywidget
import traitlets
import torch.nn as nn

from odyssey.viz.build import graph_to_module
from odyssey.viz.extract import module_to_graph
from odyssey.viz.graph import GraphSpec
from odyssey.viz.ops import list_ops, validate_graph_shapes

_STATIC = Path(__file__).resolve().parent / "static"


def _load_static(name: str) -> str:
    return (_STATIC / name).read_text(encoding="utf-8")


def _palette_payload() -> list[dict[str, Any]]:
    out = []
    for op in list_ops(palette_only=True):
        out.append({
            "name": op.name,
            "label": op.label,
            "category": op.category,
            "params": [
                {
                    "name": p.name,
                    "type": p.type,
                    "default": p.default,
                    "label": p.label or p.name,
                }
                for p in op.params
            ],
        })
    return out


class ArchEditor(anywidget.AnyWidget):
    """Interactive node canvas for viewing and designing model architectures."""

    _esm = _load_static("arch_editor.js")
    _css = _load_static("arch_editor.css")

    graph = traitlets.Dict(default_value={"nodes": [], "edges": [], "meta": {}}).tag(sync=True)
    palette = traitlets.List(default_value=[]).tag(sync=True)
    status_message = traitlets.Unicode("").tag(sync=True)
    status_kind = traitlets.Unicode("").tag(sync=True)
    validate_request = traitlets.Float(0.0).tag(sync=True)

    def __init__(self, graph: GraphSpec | dict | None = None, **kwargs: Any):
        spec = graph if isinstance(graph, GraphSpec) else GraphSpec.from_dict(graph or {})
        super().__init__(
            graph=spec.to_dict(),
            palette=_palette_payload(),
            **kwargs,
        )
        self.observe(self._on_validate, names=["validate_request"])

    @classmethod
    def from_module(
        cls,
        module: nn.Module,
        *,
        title: str | None = None,
        input_shape: tuple[int, ...] | None = None,
    ) -> ArchEditor:
        spec = module_to_graph(module, title=title, input_shape=input_shape)
        return cls(spec)

    @classmethod
    def from_json(cls, text: str) -> ArchEditor:
        return cls(GraphSpec.from_json(text))

    def get_graph(self) -> GraphSpec:
        return GraphSpec.from_dict(self.graph)

    def set_graph(self, graph: GraphSpec) -> None:
        self.graph = graph.to_dict()

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.get_graph().to_json(indent=indent)

    def validate(self) -> list[str]:
        return validate_graph_shapes(self.get_graph())

    def build(self, *, validate: bool = True) -> nn.Module:
        return graph_to_module(self.get_graph(), validate=validate)

    def _on_validate(self, change: dict) -> None:
        if not change.get("new"):
            return
        errors = self.validate()
        if errors:
            self.status_message = errors[0] if len(errors) == 1 else f"{len(errors)} shape errors"
            self.status_kind = "err"
        else:
            self.status_message = "Shape validation passed."
            self.status_kind = "ok"
