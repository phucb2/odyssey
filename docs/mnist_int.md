# MNIST integer image dataset

Data generator for CLIP int-to-text training. Use when you need multi-digit images labeled by integer.

## Purpose

Builds a processed dataset of unique `(integer, image)` pairs for `clip.py`. Each image is a horizontal strip of MNIST digit glyphs for that integer, scaled into a square canvas.

Source module: `src/odyssey/experiments/mnist_int.py`.

## Output layout

```
datasets/mnist_int/processed/
├── images/
│   ├── 0.png
│   ├── 42.png
│   └── ...
└── manifest.json
```

- PNG filenames are the integer labels.
- `manifest.json` records generation settings and the sampled label list.

Raw MNIST is downloaded separately under `datasets/mnist/raw/`.

## Generate

```bash
uv run python -m odyssey.experiments.mnist_int --generate --n-samples 2000 --seed 0
```

Demo (single random integer image):

```bash
uv run python -m odyssey.experiments.mnist_int --demo-out runs/odyssey/mnist_int_demo.png
```

## API

| Function | Role |
|----------|------|
| `number_to_image(n, ...)` | Compose one `(1, size, size)` tensor for integer `n` |
| `generate_dataset(...)` | Sample unique ints in `0..max_value`, write PNGs + manifest |
| `load_dataset(root=None)` | Load labels, paths, and optional manifest |

Defaults: `n_samples=2000`, `max_value=9999`, `size=64`, `spacing=2`, `margin=4`, `seed=0`.

## Used by

`clip.py` reads this dataset via `DataConfig.dataset = "mnist_int"` and `dataset_processed_dir("mnist_int")`.
