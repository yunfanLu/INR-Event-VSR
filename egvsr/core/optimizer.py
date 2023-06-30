from importlib import import_module

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam, SGD


class Optimizer:
    def __init__(self, config, model):
        # 1. create optimizer
        if config.NAME == "Adam":
            self.optimizer = Adam(model.parameters(), lr=config.LR)
        elif config.NAME == "SGD":
            self.optimizer = SGD(model.parameters(), lr=config.LR)
        else:
            raise ValueError(f"Unknown Optimizer config: {config}")
        # 2. create scheduler
        if config.LR_SCHEDULER == "multi_step":
            self.scheduler = lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=config.milestones,
                gamma=config.decay_gamma,
            )
        elif config.LR_SCHEDULER == "cosine":
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.end_epoch, eta_min=1e-8)
        elif config.LR_SCHEDULER == "cosine_w":
            self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-8)
        else:
            raise ValueError(f"Unknown Optimizer config: {config}")

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def lr_schedule(self):
        self.scheduler.step()
