import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, delta=0, verbose=False, pid=None):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.pid = pid
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(
                    f"PID_{self.pid}: EarlyStopping: Validation loss increased! Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f"PID_{self.pid}: EarlyStopping: Stopping training!")
                self.early_stop = True
