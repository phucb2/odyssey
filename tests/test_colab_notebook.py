"""Tests for the Colab artifact generator (no live Colab call)."""
from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "build_colab_notebook.py"
B64_RE = re.compile(r'ODYSSEY_B64 = "([^"]+)"')


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
