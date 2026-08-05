# Odyssey

Python library packaged with [uv](https://docs.astral.sh/uv/). Import as `odyssey`.

## Setup

```bash
uv sync
# or
make install
```

## Usage

```python
from odyssey import hello

print(hello())
```

## Make targets

| Target      | Description                          |
|-------------|--------------------------------------|
| `make run`  | Run `python -m odyssey`              |
| `make compile` | Byte-compile `src/`               |
| `make package` | Build sdist + wheel into `dist/`  |
| `make test` | Run pytest                           |
| `make clean`| Remove build artifacts               |

## Layout

```
src/odyssey/   # library package (installable)
tests/         # pytest suite
pyproject.toml # project + build metadata
Makefile       # common commands
docs/          # project documentation
```

See [docs/folder-structure.md](docs/folder-structure.md) for `datasets/`, `checkpoints/`, and `runs/` layout.

CLIP int-image data: [docs/mnist_int.md](docs/mnist_int.md).
