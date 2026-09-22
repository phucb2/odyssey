"""Tests for the Colab artifact generator (no live Colab call)."""
from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_colab_notebook.py"
B64_RE = re.compile(r'ODYSSEY_B64 = "([^"]+)"')


def _builder_mod():
    spec = importlib.util.spec_from_file_location("build_colab_notebook", BUILDER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_unpack(source: str, root: Path) -> str:
    patched = source.replace('Path("/content")', f"Path({str(root)!r})")
    root.mkdir(parents=True, exist_ok=True)
    script = root / "_unpack_once.py"
    script.write_text(patched, encoding="utf-8")
    return subprocess.check_output([sys.executable, str(script)], text=True)


def _run_builder(out_dir: Path, *extra: str) -> None:
    subprocess.check_call(
        [sys.executable, str(BUILDER), "--out-dir", str(out_dir), *extra],
        cwd=REPO,
    )


def _payload_from_text(text: str) -> str:
    match = B64_RE.search(text)
    assert match, "ODYSSEY_B64 assignment not found"
    payload = match.group(1)
    assert "\n" not in payload
    return payload


def _nb_text(nb: dict) -> str:
    chunks = []
    for cell in nb.get("cells") or []:
        src = cell.get("source", "")
        chunks.append("".join(src) if isinstance(src, list) else str(src))
    return "\n".join(chunks)


def test_builder_embeds_importable_odyssey(tmp_path: Path) -> None:
    out = tmp_path / "nb"
    unpacked = tmp_path / "unpacked"
    _run_builder(out)

    py_path = out / "colab_odyssey.py"
    nb_path = out / "colab_odyssey.ipynb"
    assert py_path.is_file()
    assert nb_path.is_file()

    py_text = py_path.read_text(encoding="utf-8")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        assert cell.get("id"), "nbformat 4.5 cells require an id"
    py_b64 = _payload_from_text(py_text)
    nb_b64 = _payload_from_text(_nb_text(nb))
    assert py_b64 == nb_b64
    assert py_b64

    raw = base64.b64decode(py_b64)
    unpacked.mkdir()
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
        tf.extractall(unpacked, filter="data")
    assert (unpacked / "odyssey" / "__init__.py").is_file()

    greeting = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from odyssey import hello; print(hello())",
            str(unpacked),
        ],
        text=True,
    ).strip()
    assert greeting == "Hello from odyssey!"


def test_builder_preserves_user_sections(tmp_path: Path) -> None:
    out = tmp_path / "nb"
    _run_builder(out)

    py_path = out / "colab_odyssey.py"
    nb_path = out / "colab_odyssey.ipynb"
    marker = "# USER_MARKER_4242"
    py_text = py_path.read_text(encoding="utf-8")
    py_path.write_text(py_text + f"\n{marker}\nprint('user')\n", encoding="utf-8")

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    nb["cells"].append(
        {
            "cell_type": "code",
            "id": "user-marker-4242",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [f"{marker}\n"],
        }
    )
    nb_path.write_text(json.dumps(nb), encoding="utf-8")

    _run_builder(out)
    assert marker in py_path.read_text(encoding="utf-8")
    rebuilt = json.loads(nb_path.read_text(encoding="utf-8"))
    tail = "".join(
        "".join(c.get("source", [])) if isinstance(c.get("source"), list) else str(c.get("source", ""))
        for c in rebuilt["cells"]
    )
    assert marker in tail


def test_update_payload_in_place(tmp_path: Path) -> None:
    out = tmp_path / "nb"
    _run_builder(out)
    py_path = out / "colab_odyssey.py"
    nb_path = out / "colab_odyssey.ipynb"
    marker = "# USER_KEEP_UPDATE"
    py_path.write_text(py_path.read_text(encoding="utf-8") + f"\n{marker}\n", encoding="utf-8")
    stomped, n = B64_RE.subn(r'ODYSSEY_B64 = "AAA"', py_path.read_text(encoding="utf-8"), count=1)
    assert n == 1
    py_path.write_text(stomped, encoding="utf-8")

    subprocess.check_call(
        [sys.executable, str(REPO / "scripts" / "update_colab_payload.py"), "--out-dir", str(out)],
        cwd=REPO,
    )
    updated = py_path.read_text(encoding="utf-8")
    assert marker in updated
    payload = _payload_from_text(updated)
    assert payload != "AAA"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert _payload_from_text(_nb_text(nb)) == payload


def test_unpack_and_extras_source_contain_cache_markers() -> None:
    mod = _builder_mod()
    unpack = mod.unpack_source("abc")
    extras = mod.extras_source()
    assert ".odyssey_b64.sha256" in unpack
    assert "sha256" in unpack
    assert "Cached Odyssey at" in unpack
    assert ".odyssey_extras.txt" in extras
    assert "Cached extras" in extras


def test_unpack_cache_hit_skips_extract(tmp_path: Path) -> None:
    mod = _builder_mod()
    src = tmp_path / "pkg" / "odyssey"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("hello = lambda: 'from-cache-pkg'\n", encoding="utf-8")
    payload = mod.pack_odyssey(src)
    root = tmp_path / "content"
    source = mod.unpack_source(payload)

    first = _run_unpack(source, root)
    assert "Unpacked Odyssey to" in first
    init = root / "odyssey" / "__init__.py"
    assert init.is_file()
    marker = root / ".odyssey_b64.sha256"
    assert marker.is_file()
    os.utime(init, (1_000_000_000, 1_000_000_000))

    second = _run_unpack(source, root)
    assert "Cached Odyssey at" in second
    assert init.stat().st_mtime == 1_000_000_000

    greeting = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "from odyssey import hello; print(hello())",
            str(root),
        ],
        text=True,
    ).strip()
    assert greeting == "from-cache-pkg"


def test_unpack_cache_miss_on_payload_change(tmp_path: Path) -> None:
    mod = _builder_mod()
    src_a = tmp_path / "pkg_a" / "odyssey"
    src_a.mkdir(parents=True)
    (src_a / "__init__.py").write_text("TAG = 'a'\n", encoding="utf-8")
    src_b = tmp_path / "pkg_b" / "odyssey"
    src_b.mkdir(parents=True)
    (src_b / "__init__.py").write_text("TAG = 'b'\n", encoding="utf-8")
    root = tmp_path / "content"

    _run_unpack(mod.unpack_source(mod.pack_odyssey(src_a)), root)
    assert (root / "odyssey" / "__init__.py").read_text(encoding="utf-8") == "TAG = 'a'\n"

    out = _run_unpack(mod.unpack_source(mod.pack_odyssey(src_b)), root)
    assert "Unpacked Odyssey to" in out
    assert (root / "odyssey" / "__init__.py").read_text(encoding="utf-8") == "TAG = 'b'\n"
