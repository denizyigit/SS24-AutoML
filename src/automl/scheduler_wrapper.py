from typing import Any, Optional
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, OneCycleLR


class LRSchedulerWrapper:
    def __init__(self, scheduler: LRScheduler, verbose=True, config: dict[str: Any] = None):
        self.scheduler = scheduler
        self.config = config
        self.verbose = verbose
        self.last_lr = config["learning_rate"]

        print(
            f"PID_{self.config['pid']}: Scheduler: Initialized with {self.scheduler.__class__.__name__} and learning rate: {config['learning_rate']}")

    def step(self, epoch: Optional[int] = None):
        self.scheduler.step(epoch)

        current_lr = self.get_last_lr()[0]

        if self.last_lr != current_lr:
            self.last_lr = current_lr

            # Print the updated  learning rate if and only if:
            # 1. The learning rate has been updated
            # 2. The verbose flag is set to True
            # 3. The epoch is not None (i.e., the scheduler is not being called on each batch but epoch, otherwise it would print too many times)
            if self.verbose and epoch is not None:
                print(
                    f"PID_{self.config['pid']}: Scheduler: Updated learning rate to : {current_lr}")

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
