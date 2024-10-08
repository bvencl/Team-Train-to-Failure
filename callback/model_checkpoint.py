import os
import numpy as np
import torch


# TODO függvényösszevonás, ez így cigány
class ModelCheckpoint:
    def __init__(self, type, verbose, path, neptune_logger, loaded_values):
        self.verbose = verbose
        self.val_loss_min = np.inf if loaded_values is None else loaded_values[0]
        self.val_acc_max = 0.0 if loaded_values is None else loaded_values[1] / 100
        self.path = path
        self.neptune_logger = neptune_logger
        self.type = type

    def __call__(self, val_loss, val_acc, model):
        if self.type == "accuracy" and val_acc > self.val_acc_max:
            self.save_checkpoint(val_acc, val_loss, model, "accuracy")
        elif self.type == "loss" and val_loss < self.val_loss_min:
            self.save_checkpoint(val_acc, val_loss, model, "loss")

    def save_checkpoint(self, val_acc, val_loss, model, criterion):
        if criterion == "accuracy":
            if self.verbose:
                print(f"validation accuracy increased ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model...")
            self.val_acc_max = val_acc

        elif criterion == "loss":
            if self.verbose:
                print(f"validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
            self.val_loss_min = val_loss

        torch.save(model.state_dict(), self.path)

        if self.neptune_logger is not None:
            self.neptune_logger.run["training/checkpoint_loss"].append(val_loss)
            self.neptune_logger.run["training/checkpoint_accuracy"].append(val_acc)
