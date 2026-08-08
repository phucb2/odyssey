# MNIST integer image dataset

Data generator for CLIP int-to-text training. Use when building or regenerating `datasets/mnist-int/processed/`.

## Purpose

Builds a processed dataset of `(integer, image)` pairs for `clip.py`. Each image is a horizontal strip of MNIST digit glyphs for that integer, scaled into a square canvas. Extra single-digit variants teach short captions (`"one"`, …).

Source module: `src/odyssey/experiments/mnist_int.py`.

## Output layout

```
datasets/mnist_int/processed/
├── images/
│   ├── 0.png
│   ├── 0_0.png
│   ├── 42.png
│   └── ...
└── manifest.json
```

- `{n}.png` — unique integer from the main sample.
- `{d}_{i}.png` — extra single-digit sample for digit `d` (variant `i`).
- `manifest.json` records generation settings and file lists.

Raw MNIST is downloaded separately under `datasets/mnist/raw/`.

## Generate

```bash
uv run python -m odyssey.experiments.mnist_int --generate --n-samples 2000 --n-single-per-digit 50 --seed 0
```

Demo (single random integer image):

```bash
uv run python -m odyssey.experiments.mnist_int --demo-out runs/odyssey/mnist_int_demo.png
```

## API

| Function | Role |
|----------|------|
| `number_to_image(n, ...)` | Compose one `(1, size, size)` tensor for integer `n` |
| `generate_dataset(...)` | Sample unique ints + optional single-digit oversampling |
| `parse_image_label(path)` | Label from stem (`7_3.png` → `7`) |
| `load_dataset(root=None)` | Load labels, paths, and optional manifest |

Defaults: `n_samples=2000`, `n_single_per_digit=50`, `max_value=9999`, `size=64`, `spacing=2`, `margin=4`, `seed=0`.

## Used by

`clip.py` reads this dataset via `DataConfig.dataset = "mnist_int"` and `dataset_processed_dir("mnist_int")`.
