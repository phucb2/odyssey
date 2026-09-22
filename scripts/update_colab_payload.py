#!/usr/bin/env python3
"""Refresh ODYSSEY_B64 in existing Colab artifacts.

Locates `ODYSSEY_B64 = "..."` in notebooks/colab_odyssey.py and .ipynb and
replaces it with a fresh pack of src/odyssey. Leaves the rest of the files
untouched (training cells, sentinels, comments).

  make update
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from build_colab_notebook import (  # noqa: E402
    DEFAULT_OUT_DIR,
    SRC_ODYSSEY,
    _as_nb_source,
    cell_text,
    pack_odyssey,
)

B64_ASSIGN_RE = re.compile(r'(ODYSSEY_B64 = ")[^"]*(")')


def replace_payload(text: str, payload: str, *, path: Path) -> str:
    new, n = B64_ASSIGN_RE.subn(rf"\1{payload}\2", text, count=1)
    if n != 1:
        raise SystemExit(f"{path}: expected 1 ODYSSEY_B64 assignment, found {n}")
    return new


def update_py(path: Path, payload: str) -> None:
    path.write_text(replace_payload(path.read_text(encoding="utf-8"), payload, path=path), encoding="utf-8")


def update_nb(path: Path, payload: str) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    hits = 0
    for cell in nb.get("cells") or []:
        src = cell_text(cell)
        if "ODYSSEY_B64" not in src:
            continue
        cell["source"] = _as_nb_source(replace_payload(src, payload, path=path))
        hits += 1
    if hits != 1:
        raise SystemExit(f"{path}: expected ODYSSEY_B64 in 1 cell, found {hits}")
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--src", type=Path, default=SRC_ODYSSEY)
    args = parser.parse_args(argv)
    out_dir = args.out_dir.resolve()
    payload = pack_odyssey(args.src.resolve())
    py_path = out_dir / "colab_odyssey.py"
    nb_path = out_dir / "colab_odyssey.ipynb"
    if not py_path.is_file() or not nb_path.is_file():
        raise SystemExit(f"missing Colab artifacts in {out_dir}; run `make colab-nb` first")
    update_py(py_path, payload)
    update_nb(nb_path, payload)
    print(f"Updated {py_path} ({len(payload)} char payload)")
    print(f"Updated {nb_path}")


if __name__ == "__main__":
    main()
