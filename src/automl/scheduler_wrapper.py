from typing import Any, Optional
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, OneCycleLR


class LRSchedulerWrapper:
    def __init__(self, scheduler: LRScheduler, verbose=True, config: dict[str: Any] = None):
        self.scheduler = scheduler
        self.config = config
        self.verbose = verbose

        print(
            f"PID_{self.config['pid']}: Scheduler: Initialized with {self.scheduler.__class__.__name__} and learning rate: {config['learning_rate']}")

    def step(self, epoch: Optional[int] = None):
        self.scheduler.step(epoch)

        current_lr = self.get_last_lr()[0]
        if self.verbose and current_lr != self.config["learning_rate"]:
            print(
                f"PID_{self.config['pid']}: Scheduler: Updated learning rate to : {current_lr}")

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
