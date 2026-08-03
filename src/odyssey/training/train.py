"""Build learner and run training CLI."""
import hydra
from omegaconf import DictConfig

from odyssey.experiments import build_learner
from odyssey.tracking.diagnostics import describe_model
from odyssey.training.callbacks import ActivationHistCB, LRFind, ProgressCB
from odyssey.training.config import TrainConfig, train_config_from_hydra
from odyssey.training.cuda import enable_cuda_speed_settings

enable_cuda_speed_settings()


def _set_progress_plot(cbs, plot: bool):
    return [ProgressCB(plot=plot) if isinstance(cb, ProgressCB) else cb for cb in cbs]


def run_training(cfg: TrainConfig) -> None:
    learn = build_learner(cfg, plot_progress=cfg.mode == "train" or cfg.analysis.act_hist)
    print(describe_model(learn.model))

    try:
        if cfg.mode == "train":
            if cfg.act_hist:
                learn.cbs = learn.cbs + [
                    ActivationHistCB(save_dir=cfg.act_hist_dir),
                ]
            learn.fit(cfg.epochs)
            return

        ana = cfg.analysis
        if ana.lr_finder:
            learn.cbs = _set_progress_plot(learn.cbs, plot=False)
            learn.lr_find.run(learn)

        if ana.act_hist:
            learn.cbs = learn.cbs + [
                ActivationHistCB(
                    save_dir=ana.act_hist_dir,
                    n_batches=ana.act_hist_batches,
                ),
            ]
            learn.cbs = _set_progress_plot(learn.cbs, plot=True)
            learn.fit(ana.act_hist_epochs)
    finally:
        if getattr(learn, "dls", None) is not None:
            learn.dls.shutdown()


@hydra.main(version_base="1.3", config_path="pkg://odyssey.conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(train_config_from_hydra(cfg))
