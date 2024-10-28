import torch
import copy
import os
from utils.validate_model import validate_model


class Trainer:
    def __init__(self, config, criterion, optimizer, train_loader, val_loader, test_loader, lr_scheduler, callbacks, model):
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.callbacks = callbacks
        self.model = model
        self.model_path = config.paths.model_path
        self.model_name = config.paths.model_name
        self.n_epochs = self.config.trainer.n_epochs

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.neptune_namespace = None
        self.neptune_logger = True if self.config.callbacks.neptune_logger else False
        if self.neptune_logger:
            neptune = self.callbacks["neptune_logger"]
            self.neptune_namespace = neptune.run[neptune.logger.base_namespace]

        self.checkpoint = None
        self.model_checkpoint = True if self.config.callbacks.model_checkpoint else False
        if self.model_checkpoint:
            self.checkpoint = self.callbacks["model_checkpoint"]

#! -------------------------------------------------------------------------------------------------------------------------------------

    def train(self):
        if self.neptune_logger:
            self.neptune_namespace["lr"].append(float(self.lr_scheduler.get_lr()[-1]))

        for epoch in range(self.n_epochs):
            self.model.train()
            running_loss, correct_train = 0.0, 0

            for i, (inputs, labels) in enumerate(self.train_loader):    
                inputs, labels = inputs.to(self.device), labels.to(self.device)            
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.clone())

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                correct_train += (predicted == labels).sum().item()

                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, i * len(inputs), len(self.train_loader.dataset), 100. * i / len(self.train_loader), loss.item()))

            train_loss = running_loss / len(self.train_loader)
            train_acc = 100. * correct_train / len(self.train_loader.dataset)
            
            val_loss, val_acc = validate_model(model=self.model,data_loader=self.val_loader, criterion=self.criterion)

            if self.neptune_logger:
                self.neptune_namespace["train_acc"].append(train_acc)
                self.neptune_namespace["train_loss"].append(loss.item())
                self.neptune_namespace["val_acc"].append(val_acc)
                self.neptune_namespace["val_loss"].append(val_loss)
            
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            if self.model_checkpoint:
                self.checkpoint(val_loss, val_acc, self.model, self.neptune_namespace)

            print(
                f"Epoch {epoch + 1}/{self.n_epochs} - Train loss: {train_loss[-1]:.4f},"
                f"Train accuracy: {train_acc:.2f}%, Val loss: {val_loss:.4f},"
                f"Val accuracy: {val_acc:.2f}%")
