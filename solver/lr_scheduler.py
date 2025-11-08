from bisect import bisect_right
from math import cos, pi

from torch.optim.lr_scheduler import _LRScheduler

# 学习率的变化过程为：初始极小，在warmup阶段增长后作为目标基础学习率，之后根据设定的调整策略逐步减小
class LRSchedulerWithWarmup(_LRScheduler):
    """
    自定义学习率调度器（带预热阶段和多模式调整）
    继承自PyTorch的_LRScheduler基类。
    """
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        mode="step",
        warmup_factor=1.0 / 3,
        warmup_epochs=10,
        warmup_method="linear",
        total_epochs=100,
        target_lr=0,
        power=0.9,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode = mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # 1.预热阶段，在训练开始时逐步增加学习率来提升训练稳定性​​
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        # 2.预热阶段结束，进入正式调整阶段
        # 阶梯下降
        if self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]
        # 根据mode选择策略
        epoch_ratio = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )
        # 指数下降
        if self.mode == "exp":
            factor = epoch_ratio
            return [base_lr * self.power ** factor for base_lr in self.base_lrs]
        # 线性下降
        if self.mode == "linear":
            factor = 1 - epoch_ratio
            return [base_lr * factor for base_lr in self.base_lrs]
        # 多项式下降
        if self.mode == "poly":
            factor = 1 - epoch_ratio
            return [
                self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                for base_lr in self.base_lrs
            ]
        # 余弦退火
        if self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * epoch_ratio))
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
        raise NotImplementedError