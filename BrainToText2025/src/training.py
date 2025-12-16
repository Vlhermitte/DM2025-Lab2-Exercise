import os
from typing import Optional
from tqdm import tqdm
from config import Configurator

import torch
from torch import nn
from torch.utils.data import DataLoader


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss does not improve.
    """
    def __init__(self, patience=5, min_delta=1e-3, path="best_model.pt"):
        """
        patience: epochs to wait after last improvement
        min_delta: minimum change to count as an improvement
        path: where to save the best model state_dict
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best = float("inf")
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value, model):
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            if not os.path.exists(os.path.dirname(self.path)):
                os.makedirs(os.path.dirname(self.path))
            # save best weights
            torch.save(model.state_dict(), self.path)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.should_stop = True


# python
class Trainer:
    """
    Trainer class to handle training and validation of a model.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler : Optional[torch.optim.lr_scheduler.OneCycleLR] = None,
        device: torch.device = torch.device("cpu"),
        epochs: int = 100,
        early_stop: EarlyStopping = None,
        batch_first: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.batch_first = batch_first

        self.early_stop = early_stop or EarlyStopping(patience=5, min_delta=1e-3, path="./model/best_model.pt")
        self.best_val = float("inf")
        self.scaler = torch.amp.GradScaler()

    def _train_one_epoch(self, epoch: int):
        self.model.train()
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            x, x_len, y, y_len = batch
            x, x_len, y, y_len = x.to(self.device), x_len.to(self.device), y.to(self.device), y_len.to(self.device)

            # with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            # Forward
            logits, source_lengths, target_lengths, _ = self.model(x, x_len, y, y_len)
            logits = logits.float()
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            loss = self.loss_fn(
                logits,
                y.to(dtype=torch.int32),
                x_len.to(dtype=torch.int32),
                y_len.to(dtype=torch.int32)

            )

            self.optimizer.zero_grad(set_to_none=True)
            if torch.isnan(loss):
                print("NaN loss encountered. Skipping this batch.")
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Lower clip for Transformer
            self.optimizer.step()

            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    @torch.no_grad()
    def run_validation(self) -> float:
        self.model.eval()
        total_loss, total_items = 0.0, 0
        for x, x_len, y, y_len in self.val_loader:
            x, x_len, y, y_len = x.to(self.device), x_len.to(self.device), y.to(self.device), y_len.to(self.device)
            B, L = y.shape
            # with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            logits, source_lengths, target_lengths, _ = self.model(x, x_len, y, y_len)
            logits = logits.float()
            logits = torch.clamp(logits, min=-10.0, max=10.0)

            loss = self.loss_fn(
                logits,
                y.to(dtype=torch.int32),
                x_len.to(dtype=torch.int32),
                y_len.to(dtype=torch.int32)

            )
            total_loss += loss.item() * B
            total_items += B
        avg_loss = total_loss / total_items
        return avg_loss

    def run(self) -> None:
        """
        Train the model with early stopping based on validation loss.
        """
        for epoch in range(self.epochs):
            self._train_one_epoch(epoch)

            # Validation & early stopping
            val_loss = self.run_validation()
            improved = val_loss < self.best_val - 1e-9
            if improved:
                self.best_val = val_loss
            self.early_stop.step(val_loss, self.model)

            print(f"[Epoch {epoch+1}] val_loss={val_loss:.4f} best_val={self.best_val:.4f} "
                  f"patience_used={self.early_stop.bad_epochs}/{self.early_stop.patience}")

            if self.early_stop.should_stop:
                print("Early stopping triggered. Reloading best weights and exiting training.")
                self.model.load_state_dict(torch.load(self.early_stop.path, map_location=self.device))
                break