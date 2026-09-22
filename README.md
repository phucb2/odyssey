# Odyssey

Project setup and how to run locally and on Colab. Import as `odyssey`. Packaged with [uv](https://docs.astral.sh/uv/).

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

## Colab

Self-contained Colab bootstrap: `make colab-nb` writes `notebooks/colab_odyssey.py` (CLI) and `notebooks/colab_odyssey.ipynb` (UI). After package edits, `make update` replaces only the `ODYSSEY_B64` payload and leaves experiment cells as they are. Colab CLI sends only that file, not the repo.

```bash
# Pin jupyter-kernel-client: google-colab-cli 0.6.x calls KernelClient, removed in 1.x
uv tool install --force --with 'jupyter-kernel-client==0.15.0' google-colab-cli
gcloud auth application-default login --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory

make colab-nb   # first time
make update     # after src/odyssey edits
colab new -s odyssey          # add --gpu T4 when you add training
colab exec -s odyssey -f notebooks/colab_odyssey.py --timeout 600
# one-shot: colab run --gpu T4 --timeout 600 notebooks/colab_odyssey.py
# optional UI notebook: colab exec -s odyssey -f notebooks/colab_odyssey.ipynb --timeout 600
colab stop -s odyssey
```

`--timeout 600` is required (CLI default is 30s). Always `colab stop` when done.

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
