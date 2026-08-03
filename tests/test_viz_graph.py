"""Tests for odyssey.viz graph IR, extract/build, and editor API."""
import torch
import torch.nn as nn

from odyssey.models import create_cnn_model, create_res_model
from odyssey.viz import ArchEditor, GraphSpec, graph_to_module, module_to_graph, validate_graph_shapes
from odyssey.viz.ops import infer_shape


def test_graph_json_roundtrip():
    g = GraphSpec()
    n1 = g.add_node("input", params={"shape": [1, 28, 28]})
    n2 = g.add_node("reshape", params={"shape": [1, 28, 28]})
    g.add_edge(n1.id, n2.id)
    restored = GraphSpec.from_json(g.to_json())
    assert len(restored.nodes) == 2
    assert restored.edges[0].source == n1.id


def test_extract_cnn_has_linear_chain():
    model = create_cnn_model()
    graph = module_to_graph(model, input_shape=(1, 1, 28, 28))
    assert graph.nodes[0].op == "input"
    ops = [n.op for n in graph.nodes]
    assert "reshape" in ops
    assert "conv" in ops or "conv2d" in ops
    assert "linear" in ops
    assert len(graph.edges) >= len(graph.nodes) - 1


def test_build_from_extracted_cnn_runs_forward():
    model = create_cnn_model()
    graph = module_to_graph(model, input_shape=(1, 784))
    rebuilt = graph_to_module(graph, validate=False)
    x = torch.zeros(2, 784)
    out = rebuilt(x)
    assert out.shape == (2, 10)


def test_resblock_roundtrip():
    graph = GraphSpec(meta={"input_shape": (1, 8, 28, 28)})
    n0 = graph.add_node("input", params={"shape": [8, 28, 28]})
    n1 = graph.add_node("resblock", params={"in_ch": 8, "out_ch": 16})
    graph.add_edge(n0.id, n1.id)
    model = graph_to_module(graph)
    x = torch.randn(1, 8, 28, 28)
    y = model(x)
    assert y.shape == (1, 16, 28, 28)


def test_validate_detects_bad_linear():
    graph = GraphSpec(meta={"input_shape": (1, 10)})
    n0 = graph.add_node("input", params={"shape": [10]})
    n1 = graph.add_node("linear", params={"in_features": 784, "out_features": 10})
    graph.add_edge(n0.id, n1.id)
    errors = validate_graph_shapes(graph)
    assert errors


def test_infer_shape_conv():
    out = infer_shape("conv2d", (1, 1, 28, 28), {"in_ch": 1, "out_ch": 8, "kernel_size": 5, "padding": 2})
    assert out == (1, 8, 28, 28)


def test_arch_editor_from_module_and_build():
    editor = ArchEditor.from_module(create_res_model(wide=False), input_shape=(1, 1, 28, 28))
    spec = editor.get_graph()
    assert spec.meta["title"]
    assert len(spec.nodes) > 3
    json_text = editor.to_json()
    assert "nodes" in json_text
    rebuilt = editor.build(validate=False)
    assert isinstance(rebuilt, nn.Module)


def test_arch_editor_validate_ok_on_simple_graph():
    editor = ArchEditor()
    g = GraphSpec(meta={"input_shape": (1, 1, 28, 28)})
    n0 = g.add_node("input", params={"shape": [1, 28, 28]})
    n1 = g.add_node("flatten", params={})
    g.add_edge(n0.id, n1.id)
    editor.set_graph(g)
    assert editor.validate() == []
