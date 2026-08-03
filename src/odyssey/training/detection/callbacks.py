"""DETR-specific training callbacks."""
from odyssey.training.callback import Callback
from odyssey.training.callbacks import LossDictMetricsCB, MetricsCB
from odyssey.training.detection.metrics import DetectionMapMetric


class DetrTrainCB(Callback):
    def predict(self, learn):
        return learn.model(learn.batch[0])

    def get_loss(self, learn):
        loss, loss_dict = learn.loss_func(learn.preds, learn.batch[1])
        learn.loss_dict = loss_dict
        return loss

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad(set_to_none=True)


class DetrLossMetricsCB(LossDictMetricsCB):
    """Track DETR criterion loss components."""

    def __init__(self):
        super().__init__(("loss_ce", "loss_bbox", "loss_giou"))


class DetrMapMetricsCB(Callback):
    """Accumulate validation predictions and log mAP@0.5 / mAP@0.5:0.95."""

    order = MetricsCB.order - 1

    def __init__(self, num_classes: int, score_threshold: float = 0.05):
        self.metric = DetectionMapMetric(
            num_classes=num_classes,
            score_threshold=score_threshold,
        )

    def before_epoch(self, learn):
        if not learn.training:
            self.metric.reset()

    def after_batch(self, learn):
        if learn.training or not isinstance(learn.preds, dict):
            return
        self.metric.update(learn.preds, learn.batch[1])

    def after_epoch(self, learn):
        if learn.model.training:
            return
        learn.epoch_metrics = getattr(learn, "epoch_metrics", {})
        for key, value in self.metric.compute().items():
            learn.epoch_metrics[key] = f"{value:.4f}"
