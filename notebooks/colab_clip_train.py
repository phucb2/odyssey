# Colab: run Odyssey CLIP training (existing code)
# Runtime → Change runtime type → GPU

# !pip install -q uv
# !git clone https://github.com/phucb2/odyssey.git
# %cd odyssey
# !uv sync

from odyssey.experiments.mnist_int import generate_dataset
from odyssey.experiments.clip import compose_clip_config, train

# Generate MNIST-int once (skip if already present)
generate_dataset(n_samples=2000, seed=0)

# Hydra-style overrides — same as CLI: python -m odyssey.experiments.clip fit.epochs=5
cfg = compose_clip_config(
    [
        "fit.epochs=5",
        "fit.lr=3e-4",
        "data.batch_size=64",
    ]
)
learn = train(cfg, plot_progress=False)
print("checkpoint:", learn.checkpoint_path)
