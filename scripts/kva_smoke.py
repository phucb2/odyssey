"""Minimal kva exec smoke test. Stdlib only."""

import platform
import sys

print("kva-ok")
print("python", sys.version.split()[0], flush=True)
print("platform", platform.platform(), flush=True)
