#!/usr/bin/env python3
"""Embed src/odyssey into self-contained Colab artifacts.

Run from anywhere: writes notebooks/colab_odyssey.py and .ipynb.
Regenerate after package edits with `make colab-nb`.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import tarfile
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ODYSSEY = REPO_ROOT / "src" / "odyssey"
DEFAULT_OUT_DIR = REPO_ROOT / "notebooks"

PY_SENTINEL = "# --- odyssey:user-cells ---"
NB_SENTINEL = "<!-- odyssey:user-cells -->"

_COLAB_EXTRAS = (
    "fastcore>=1.7.0",
    "fastai>=2.7.0",
    "rich>=13.0",
    "torcheval>=0.0.7",
    "einops>=0.8.2",
    "hydra-core>=1.3.0",
    "omegaconf>=2.3.1",
    "medmnist>=3.0.2",
    "anywidget>=0.9.0",
)

HEADER_MD = """\
# Odyssey Colab bootstrap

Self-contained Colab bootstrap. Regenerates Odyssey from an embedded base64 \
payload so Colab CLI does not need the git repo.

Regenerate after package edits: `make colab-nb`

## CLI (preferred: script)

```bash
uv tool install --force --with 'jupyter-kernel-client==0.15.0' google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

make colab-nb
colab new -s odyssey          # add --gpu T4 when you add training
colab exec -s odyssey -f notebooks/colab_odyssey.py --timeout 600
# one-shot: colab run --gpu T4 --timeout 600 notebooks/colab_odyssey.py
colab stop -s odyssey
```

`--timeout 600` is required (CLI default is 30s). Always `colab stop` when done.

## Vast.ai (`kva exec`)

SSH uses `-i ~/.ssh/id_ed25519`. Register that pubkey with `vastai create ssh-key` before creating the instance.

```bash
kva sessions
kva exec -s INSTANCE_ID -f notebooks/colab_odyssey.py
# long jobs: add --detach  (logs: runs/kva/ locally, /workspace/odyssey/runs/kva/ on the host)
kva upload -s INSTANCE_ID ./datasets/coco128/raw/coco128.zip
kva stop -s INSTANCE_ID
```

## Notebook (Colab UI or CLI)

This `.ipynb` is self-contained. Optional: \
`colab exec -s odyssey -f notebooks/colab_odyssey.ipynb --timeout 600` \
writes `notebooks/colab_odyssey_output.ipynb`.
"""

HEADER_PY = '''\
"""Odyssey Colab bootstrap.

Self-contained Colab bootstrap. Regenerates Odyssey from an embedded base64
payload so Colab CLI does not need the git repo.

Regenerate after package edits: `make colab-nb`

CLI (preferred):
  uv tool install --force --with 'jupyter-kernel-client==0.15.0' google-colab-cli
  make colab-nb
  colab new -s odyssey          # add --gpu T4 when you add training
  colab exec -s odyssey -f notebooks/colab_odyssey.py --timeout 600
  # one-shot: colab run --gpu T4 --timeout 600 notebooks/colab_odyssey.py
  colab stop -s odyssey

`--timeout 600` is required (CLI default is 30s). Always `colab stop` when done.

Vast.ai:
  kva new                         # cheap GPU; warns about $/hr then asks y/N
  kva new --size medium           # 20GB+; still price-prompts
  kva new --size large            # 40GB+; still price-prompts
  kva exec -s INSTANCE_ID -f notebooks/colab_odyssey.py
  # long jobs: kva exec -s INSTANCE_ID -f notebooks/colab_odyssey.py --detach
  kva sessions
  kva stop -s INSTANCE_ID
  kva upload -s INSTANCE_ID ./datasets/coco128/raw/coco128.zip
  kva download -s INSTANCE_ID /workspace/runs ./runs
  # SSH key: ~/.ssh/id_ed25519 (vastai create ssh-key ~/.ssh/id_ed25519.pub)
"""
'''

DEFAULT_USER_PY = """
# Add experiment / training code below this line.
"""

DEFAULT_USER_NB_CODE = "# Add experiment / training code below this line."

DEFAULT_USER_NB_MD = f"""\
{NB_SENTINEL}

Add experiment cells below.
"""


def _tar_filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    parts = Path(tarinfo.name).parts
    if "__pycache__" in parts:
        return None
    if tarinfo.name.endswith((".pyc", ".pyo")):
        return None
    return tarinfo


def pack_odyssey(src: Path = SRC_ODYSSEY) -> str:
    "gzip tar of src/odyssey as odyssey/, then base64."
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(src, arcname="odyssey", filter=_tar_filter)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def unpack_source(payload: str) -> str:
    return f'''\
# @title Unpack Odyssey
ODYSSEY_B64 = "{payload}"

import base64
import importlib
import io
import sys
import tarfile
from pathlib import Path

_root = Path("/content")
_root.mkdir(parents=True, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ODYSSEY_B64)), mode="r:gz") as tf:
    tf.extractall(_root, filter="data")
_root_s = str(_root)
sys.path = [p for p in sys.path if p != _root_s]
sys.path.insert(0, _root_s)
# Same Colab kernel keeps old imports; drop them so the new tree is loaded.
for _name in list(sys.modules):
    if _name == "odyssey" or _name.startswith("odyssey."):
        del sys.modules[_name]
importlib.invalidate_caches()
print("Unpacked Odyssey to", _root / "odyssey")
'''


def extras_source() -> str:
    pkgs = ",\n    ".join(f'"{p}"' for p in _COLAB_EXTRAS)
    return f'''\
# @title Install extras
import subprocess
import sys

_EXTRAS = [
    {pkgs},
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *_EXTRAS])
print("Installed extras:", ", ".join(_EXTRAS))
'''


def sanity_source() -> str:
    return '''\
# @title Sanity check
import torch
from odyssey import __version__, hello
from odyssey.training.learner import Learner

print(hello())
print("odyssey", __version__)
print("cuda", torch.cuda.is_available())
print("Learner", Learner)
'''


def _as_nb_source(text: str) -> list[str]:
    if not text:
        return []
    parts = text.split("\n")
    if parts and parts[-1] == "":
        parts = parts[:-1]
        return [p + "\n" for p in parts]
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def _cell_id() -> str:
    return uuid.uuid4().hex[:12]


def _ensure_cell_id(cell: dict) -> dict:
    if not cell.get("id"):
        cell["id"] = _cell_id()
    return cell


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": _cell_id(),
        "metadata": {},
        "source": _as_nb_source(source),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _cell_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _as_nb_source(source),
    }


def cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        return "".join(src)
    return str(src)


def read_user_py(path: Path) -> str:
    if not path.is_file():
        return DEFAULT_USER_PY
    text = path.read_text(encoding="utf-8")
    idx = text.find(PY_SENTINEL)
    if idx < 0:
        return DEFAULT_USER_PY
    return text[idx + len(PY_SENTINEL) :]


def read_user_nb_cells(path: Path) -> list[dict]:
    fallback = [code_cell(DEFAULT_USER_NB_CODE)]
    if not path.is_file():
        return fallback
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return fallback
    cells = nb.get("cells") or []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown" and NB_SENTINEL in cell_text(cell):
            tail = cells[i + 1 :]
            return [_ensure_cell_id(c) for c in tail] if tail else fallback
    return fallback


def write_py(path: Path, payload: str, user: str) -> None:
    user_block = user if user.startswith("\n") else "\n" + user
    body = "\n".join(
        [
            HEADER_PY.rstrip(),
            "",
            unpack_source(payload).rstrip(),
            "",
            extras_source().rstrip(),
            "",
            sanity_source().rstrip(),
            "",
            PY_SENTINEL + user_block,
        ]
    )
    if not body.endswith("\n"):
        body += "\n"
    path.write_text(body, encoding="utf-8")


def write_nb(path: Path, payload: str, user_cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            md_cell(HEADER_MD),
            code_cell(unpack_source(payload)),
            code_cell(extras_source()),
            code_cell(sanity_source()),
            md_cell(DEFAULT_USER_NB_MD),
            *user_cells,
        ],
    }
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def build(out_dir: Path, src: Path = SRC_ODYSSEY) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    py_path = out_dir / "colab_odyssey.py"
    nb_path = out_dir / "colab_odyssey.ipynb"
    payload = pack_odyssey(src)
    user_py = read_user_py(py_path)
    user_nb = read_user_nb_cells(nb_path)
    write_py(py_path, payload, user_py)
    write_nb(nb_path, payload, user_nb)
    return py_path, nb_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for colab_odyssey.py and .ipynb (default: notebooks/)",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=SRC_ODYSSEY,
        help="Package directory to embed (default: src/odyssey)",
    )
    args = parser.parse_args(argv)
    py_path, nb_path = build(args.out_dir.resolve(), src=args.src.resolve())
    print(f"Wrote {py_path} ({py_path.stat().st_size} bytes)")
    print(f"Wrote {nb_path} ({nb_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
