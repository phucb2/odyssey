# Odyssey

Project setup and how to run locally, on Colab, and on Vast.ai. Import as `odyssey`. Packaged with [uv](https://docs.astral.sh/uv/).

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
| `make colab-nb` | Embed `src/odyssey` into Colab artifacts |
| `make update` | Refresh `ODYSSEY_B64` in existing Colab files |
| `make test` | Run pytest                           |
| `make clean`| Remove build artifacts               |

## Remote GPU (Colab / Vast.ai)

Getting started for `colab` and `kva`: [docs/colab-kva.md](docs/colab-kva.md).

```bash
make colab-nb
colab new -s odyssey          # add --gpu T4 when training
colab exec -s odyssey -f notebooks/colab_odyssey.py --timeout 600
colab stop -s odyssey

kva new                       # Vast.ai GPU; y/n per offer
kva exec -f notebooks/colab_odyssey.py --setup
kva stop
```

## Layout

```
src/odyssey/   # library package (installable)
scripts/       # generators (Colab embed)
notebooks/     # Colab artifacts and local notebooks
tests/         # pytest suite
pyproject.toml # project + build metadata
Makefile       # common commands
docs/          # project documentation
```

See [docs/folder-structure.md](docs/folder-structure.md) for `datasets/`, `checkpoints/`, and `runs/` layout.

CLIP int-image data: [docs/mnist_int.md](docs/mnist_int.md).

Small math lexical corpus: [docs/small_math_corpus.md](docs/small_math_corpus.md).

Plain / text-only training logs: [docs/plain-console.md](docs/plain-console.md).
