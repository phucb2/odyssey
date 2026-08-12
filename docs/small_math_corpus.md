# Small English math lexical corpus

Generated arithmetic sentences for char LM pretrain and one-shot math tuning. Use when building or regenerating `datasets/small/`.

## Purpose

Builds `(equation, sentence)` pairs for `math_lm.py`. Each equation yields several English templates (`nine plus sixteen equals twenty five`, …). Integer operands (`I`) and reduced fractions (`Q`) up to configurable bounds.

Source module: `src/odyssey/experiments/math_corpus.py`.

## Output layout

```
datasets/small/
├── corpus.txt              # all sentence variants (pretrain eval / inspection)
├── corpus_train.txt        # train-split sentences only (Step 1 pretrain)
├── equations.jsonl         # all equations + sentence lists
├── equations_train.jsonl   # train split (Step 2 tune)
├── equations_holdout.jsonl # held-out one-shot eval
└── manifest.json
```

## Generate

```bash
uv run python -m odyssey.experiments.math_corpus --generate --n-equations 4000 --max-int 20 --max-den 8 --seed 0
```

## Sync (DVC)

Dataset blobs live in the GCS remote (`gs://odyssey-dvc`). Pull after clone:

```bash
uv run dvc pull datasets/small.dvc
```

Re-upload after regenerating:

```bash
uv run dvc add datasets/small
uv run dvc push datasets/small.dvc
```

## Used by

`math_lm.py` reads this dataset via `DataConfig.dataset = "small"` and `dataset_dir("small")`.
