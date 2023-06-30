from egvsr.losses.loss_factory import MixedLoss
from egvsr.losses.metric_factory import MixedMetric


def get_loss(config):
    return MixedLoss(config)


def get_metric(configs):
    return MixedMetric(configs)
