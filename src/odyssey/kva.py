"""kva — Vast.ai exec CLI (colab exec analogue)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_kva_script() -> Path:
    starts = [Path(__file__).resolve().parent, Path.cwd()]
    for start in starts:
        for parent in [start, *start.parents]:
            script = parent / "scripts" / "kva"
            if script.is_file() and (parent / "pyproject.toml").is_file():
                return script
    raise SystemExit("kva: could not find scripts/kva (run from the odyssey repo)")


def main() -> None:
    script = find_kva_script()
    os.execv("/bin/bash", ["bash", str(script), *sys.argv[1:]])


if __name__ == "__main__":
    main()
