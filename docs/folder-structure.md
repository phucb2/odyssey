# Folder structure

Reference for on-disk layout used by Odyssey training. Paths are resolved from the repository root (where you run `mltrain`).

## Layout

```
├── datasets/
│   ├── cifar10/
│   │   ├── raw/          # torchvision CIFAR-10 download root
│   │   ├── processed/    # GPU-preloaded .pt cache
│   │   └── manifests/    # optional metadata
│   └── fashion-mnist/
│       ├── raw/          # torchvision FashionMNIST download root
│       ├── processed/    # GPU-preloaded .pt cache (e.g. fashion_mnist.pt)
│       └── manifests/
│   └── coco128/
│       ├── raw/          # Ultralytics coco128 zip extract (images/, labels/)
│       └── manifests/    # optional metadata
├── checkpoints/
│   └── odyssey/          # model weights (*.pt, *.pth); project name is configurable
└── runs/
    └── odyssey/          # TensorBoard logs, LR finder plots, activation diagnostics
        ├── tensorboard/
        ├── activations/
        └── lr_find.png
```

## What goes where

| Path | Purpose |
|------|---------|
| `datasets/*/raw/` | Original downloads (torchvision, manual drops) |
| `datasets/*/processed/` | Preprocessed GPU tensors cached to disk |
| `datasets/*/manifests/` | Index files, split lists, label maps |
| `checkpoints/<project>/` | Saved model checkpoints |
| `runs/<project>/` | Experiment logs and analysis artifacts |

## Defaults

- Project name: `odyssey` (`TrainConfig.project_name` or `--project-name`)
- Fashion-MNIST raw root: `datasets/fashion-mnist/raw/`
- Fashion-MNIST GPU cache: `datasets/fashion-mnist/processed/fashion_mnist.pt`
- CIFAR-10 auto-downloads to `datasets/cifar10/raw/` on first run (`mltrain model=resnet dataset=cifar10`)
- coco128 auto-downloads to `datasets/coco128/raw/` on first DETR run (`mltrain task=detr dataset=coco128`)

## Git

`datasets/`, `checkpoints/`, and `runs/` are gitignored. Only this document and source code are tracked.
