/* Odyssey architecture node editor — anywidget frontend */
const NODE_W = 148;
const NODE_H = 56;
const PORT_R = 5;

function parseGraph(raw) {
  if (!raw) return { nodes: [], edges: [], meta: {} };
  if (typeof raw === "string") {
    try { return JSON.parse(raw); } catch { return { nodes: [], edges: [], meta: {} }; }
  }
  return raw;
}

function cloneGraph(g) {
  return JSON.parse(JSON.stringify(g));
}

function uid(prefix) {
  return `${prefix}_${Math.random().toString(16).slice(2, 10)}`;
}

function nodeLabel(n) {
  return n.label || n.op;
}

function nodeSubtitle(n) {
  const p = n.params || {};
  const keys = Object.keys(p).slice(0, 2);
  if (!keys.length) return "";
  return keys.map((k) => `${k}=${p[k]}`).join(" · ");
}

export default {
  render({ model, el, signal }) {
    el.classList.add("odyssey-arch-editor");

    const shell = document.createElement("div");
    shell.className = "oae-shell";
    el.appendChild(shell);

    const toolbar = document.createElement("div");
    toolbar.className = "oae-toolbar";
    toolbar.innerHTML = `
      <span class="oae-title">Architecture</span>
      <button type="button" data-act="fit">Fit</button>
      <button type="button" data-act="layout">Auto-layout</button>
      <button type="button" data-act="validate">Validate</button>
      <button type="button" data-act="clear">Clear</button>
    `;
    shell.appendChild(toolbar);

    const body = document.createElement("div");
    body.className = "oae-body";
    shell.appendChild(body);

    const palette = document.createElement("div");
    palette.className = "oae-palette";
    palette.innerHTML = `<div class="oae-section-title">Layers</div>`;
    body.appendChild(palette);

    const canvasWrap = document.createElement("div");
    canvasWrap.className = "oae-canvas-wrap";
    body.appendChild(canvasWrap);

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "oae-canvas");
    canvasWrap.appendChild(svg);

    const inspector = document.createElement("div");
    inspector.className = "oae-inspector";
    inspector.innerHTML = `<div class="oae-section-title">Inspector</div><div class="oae-inspector-body"></div>`;
    body.appendChild(inspector);

    const status = document.createElement("div");
    status.className = "oae-status";
    shell.appendChild(status);

    let graph = cloneGraph(parseGraph(model.get("graph")));
    let paletteOps = model.get("palette") || [];
    let selectedId = null;
    let drag = null;
    let wire = null;
    let pan = { x: 0, y: 0, active: false, sx: 0, sy: 0 };

    function setStatus(msg, kind = "") {
      status.textContent = msg;
      status.dataset.kind = kind;
    }

    function pushGraph() {
      model.set("graph", cloneGraph(graph));
      model.save_changes();
    }

    function renderPalette() {
      palette.querySelectorAll(".oae-palette-btn").forEach((n) => n.remove());
      paletteOps.forEach((op) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "oae-palette-btn";
        btn.textContent = op.label || op.name;
        btn.title = op.name;
        btn.addEventListener("click", () => addNode(op), { signal });
        palette.appendChild(btn);
      });
    }

    function addNode(op) {
      const id = uid("n");
      const params = {};
      (op.params || []).forEach((f) => { params[f.name] = f.default; });
      const n = {
        id,
        op: op.name,
        params,
        x: 80 + graph.nodes.length * 40,
        y: 80 + (graph.nodes.length % 3) * 70,
      };
      graph.nodes.push(n);
      selectedId = id;
      pushGraph();
      draw();
      renderInspector();
    }

    function removeSelected() {
      if (!selectedId) return;
      graph.nodes = graph.nodes.filter((n) => n.id !== selectedId);
      graph.edges = graph.edges.filter((e) => e.source !== selectedId && e.target !== selectedId);
      selectedId = null;
      pushGraph();
      draw();
      renderInspector();
    }

    function autoLayout() {
      const order = topoSort();
      order.forEach((id, i) => {
        const n = graph.nodes.find((x) => x.id === id);
        if (n) {
          n.x = 40 + i * 180;
          n.y = 90;
        }
      });
      pushGraph();
      draw();
    }

    function topoSort() {
      const ids = graph.nodes.map((n) => n.id);
      const indeg = Object.fromEntries(ids.map((id) => [id, 0]));
      graph.edges.forEach((e) => { indeg[e.target] = (indeg[e.target] || 0) + 1; });
      const q = ids.filter((id) => indeg[id] === 0);
      const out = [];
      const adj = {};
      graph.edges.forEach((e) => {
        adj[e.source] = adj[e.source] || [];
        adj[e.source].push(e.target);
      });
      while (q.length) {
        const id = q.shift();
        out.push(id);
        (adj[id] || []).forEach((t) => {
          indeg[t] -= 1;
          if (indeg[t] === 0) q.push(t);
        });
      }
      return out.length === ids.length ? out : ids;
    }

    function fitView() {
      if (!graph.nodes.length) return;
      const xs = graph.nodes.map((n) => n.x);
      const ys = graph.nodes.map((n) => n.y);
      const minX = Math.min(...xs) - 20;
      const minY = Math.min(...ys) - 20;
      pan.x = -minX + 10;
      pan.y = -minY + 10;
      draw();
    }

    function renderInspector() {
      const bodyEl = inspector.querySelector(".oae-inspector-body");
      bodyEl.innerHTML = "";
      const node = graph.nodes.find((n) => n.id === selectedId);
      if (!node) {
        bodyEl.textContent = "Select a node to edit parameters.";
        return;
      }
      const title = document.createElement("div");
      title.className = "oae-inspector-title";
      title.textContent = `${nodeLabel(node)} (${node.op})`;
      bodyEl.appendChild(title);

      const opDef = paletteOps.find((o) => o.name === node.op);
      const fields = opDef?.params || Object.keys(node.params || {}).map((k) => ({ name: k, type: "str", default: node.params[k] }));

      fields.forEach((field) => {
        const row = document.createElement("label");
        row.className = "oae-field";
        const span = document.createElement("span");
        span.textContent = field.label || field.name;
        const input = document.createElement("input");
        input.value = node.params[field.name] ?? field.default ?? "";
        input.addEventListener("change", () => {
          let val = input.value;
          if (field.type === "int") val = parseInt(val, 10) || 0;
          else if (field.type === "float") val = parseFloat(val) || 0;
          else if (field.type === "bool") val = input.value === "true";
          node.params[field.name] = val;
          pushGraph();
          draw();
        }, { signal });
        row.appendChild(span);
        row.appendChild(input);
        bodyEl.appendChild(row);
      });

      const del = document.createElement("button");
      del.type = "button";
      del.className = "oae-delete-btn";
      del.textContent = "Delete node";
      del.addEventListener("click", removeSelected, { signal });
      bodyEl.appendChild(del);
    }

    function portPos(node, side) {
      if (side === "out") return { x: node.x + NODE_W, y: node.y + NODE_H / 2 };
      return { x: node.x, y: node.y + NODE_H / 2 };
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.setAttribute("transform", `translate(${pan.x},${pan.y})`);
      svg.appendChild(g);

      graph.edges.forEach((e) => {
        const src = graph.nodes.find((n) => n.id === e.source);
        const tgt = graph.nodes.find((n) => n.id === e.target);
        if (!src || !tgt) return;
        const p0 = portPos(src, "out");
        const p1 = portPos(tgt, "in");
        const mx = (p0.x + p1.x) / 2;
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M ${p0.x} ${p0.y} C ${mx} ${p0.y}, ${mx} ${p1.y}, ${p1.x} ${p1.y}`);
        path.setAttribute("class", "oae-edge");
        path.dataset.id = e.id;
        g.appendChild(path);
      });

      if (wire) {
        const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
        path.setAttribute("d", `M ${wire.x0} ${wire.y0} L ${wire.x1} ${wire.y1}`);
        path.setAttribute("class", "oae-edge oae-edge-draft");
        g.appendChild(path);
      }

      graph.nodes.forEach((node) => {
        const ng = document.createElementNS("http://www.w3.org/2000/svg", "g");
        ng.setAttribute("class", "oae-node" + (node.id === selectedId ? " oae-node-selected" : ""));
        ng.setAttribute("transform", `translate(${node.x},${node.y})`);
        ng.dataset.id = node.id;

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("width", NODE_W);
        rect.setAttribute("height", NODE_H);
        rect.setAttribute("rx", 4);
        ng.appendChild(rect);

        const t1 = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t1.setAttribute("x", 10);
        t1.setAttribute("y", 22);
        t1.setAttribute("class", "oae-node-title");
        t1.textContent = nodeLabel(node);
        ng.appendChild(t1);

        const t2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t2.setAttribute("x", 10);
        t2.setAttribute("y", 40);
        t2.setAttribute("class", "oae-node-sub");
        t2.textContent = nodeSubtitle(node);
        ng.appendChild(t2);

        if (node.op !== "input") {
          const pin = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          pin.setAttribute("cx", 0);
          pin.setAttribute("cy", NODE_H / 2);
          pin.setAttribute("r", PORT_R);
          pin.setAttribute("class", "oae-port oae-port-in");
          ng.appendChild(pin);
        }

        const pout = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        pout.setAttribute("cx", NODE_W);
        pout.setAttribute("cy", NODE_H / 2);
        pout.setAttribute("r", PORT_R);
        pout.setAttribute("class", "oae-port oae-port-out");
        ng.appendChild(pout);

        g.appendChild(ng);
      });
    }

    function clientToGraph(evt) {
      const pt = svg.createSVGPoint();
      pt.x = evt.clientX;
      pt.y = evt.clientY;
      const ctm = svg.getScreenCTM();
      if (!ctm) return { x: 0, y: 0 };
      const loc = pt.matrixTransform(ctm.inverse());
      return { x: loc.x - pan.x, y: loc.y - pan.y };
    }

    function hitNode(x, y) {
      return graph.nodes.find((n) => x >= n.x && x <= n.x + NODE_W && y >= n.y && y <= n.y + NODE_H);
    }

    svg.addEventListener("mousedown", (evt) => {
      const { x, y } = clientToGraph(evt);
      const target = evt.target;
      const nodeG = target.closest?.(".oae-node");
      if (target.classList?.contains("oae-port-out") && nodeG) {
        const id = nodeG.dataset.id;
        const node = graph.nodes.find((n) => n.id === id);
        const p = portPos(node, "out");
        wire = { source: id, x0: p.x, y0: p.y, x1: p.x, y1: p.y };
        draw();
        return;
      }
      if (target.classList?.contains("oae-port-in") && nodeG && wire) {
        const tgt = nodeG.dataset.id;
        if (wire.source !== tgt) {
          graph.edges.push({ id: uid("e"), source: wire.source, target: tgt, source_port: "out", target_port: "in" });
          pushGraph();
        }
        wire = null;
        draw();
        return;
      }
      const node = hitNode(x, y);
      if (node) {
        selectedId = node.id;
        drag = { id: node.id, ox: x - node.x, oy: y - node.y };
        renderInspector();
        draw();
        return;
      }
      pan.active = true;
      pan.sx = evt.clientX - pan.x;
      pan.sy = evt.clientY - pan.y;
    }, { signal });

    window.addEventListener("mousemove", (evt) => {
      if (wire) {
        const { x, y } = clientToGraph(evt);
        wire.x1 = x;
        wire.y1 = y;
        draw();
      }
      if (drag) {
        const { x, y } = clientToGraph(evt);
        const node = graph.nodes.find((n) => n.id === drag.id);
        if (node) {
          node.x = x - drag.ox;
          node.y = y - drag.oy;
          draw();
        }
      }
      if (pan.active) {
        pan.x = evt.clientX - pan.sx;
        pan.y = evt.clientY - pan.sy;
        draw();
      }
    }, { signal });

    window.addEventListener("mouseup", () => {
      if (drag) {
        pushGraph();
        drag = null;
      }
      if (wire) {
        wire = null;
        draw();
      }
      pan.active = false;
    }, { signal });

    window.addEventListener("keydown", (evt) => {
      if (evt.key === "Delete" || evt.key === "Backspace") {
        if (document.activeElement?.tagName === "INPUT") return;
        removeSelected();
      }
    }, { signal });

    toolbar.addEventListener("click", (evt) => {
      const act = evt.target?.dataset?.act;
      if (act === "fit") fitView();
      if (act === "layout") autoLayout();
      if (act === "validate") {
        model.set("validate_request", Date.now());
        model.save_changes();
      }
      if (act === "clear") {
        graph = { nodes: [], edges: [], meta: graph.meta || {} };
        selectedId = null;
        pushGraph();
        draw();
        renderInspector();
      }
    }, { signal });

    model.on("change:graph", () => {
      graph = cloneGraph(parseGraph(model.get("graph")));
      draw();
      renderInspector();
    }, { signal });

    model.on("change:palette", () => {
      paletteOps = model.get("palette") || [];
      renderPalette();
    }, { signal });

    model.on("change:status_message", () => {
      setStatus(model.get("status_message") || "", model.get("status_kind") || "");
    }, { signal });

    graph = cloneGraph(parseGraph(model.get("graph")));
    paletteOps = model.get("palette") || [];
    renderPalette();
    draw();
    renderInspector();
    setStatus(model.get("status_message") || "Drag from output port to input port to connect.", "");
    fitView();
  },
};
