"""Training loop and configuration."""
from odyssey.training.config import TrainConfig, compose_train_config
from odyssey.training.learner import Learner, MomentumLearner

__all__ = ["Learner", "MomentumLearner", "TrainConfig", "compose_train_config"]
