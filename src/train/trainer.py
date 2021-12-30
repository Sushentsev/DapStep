from typing import Optional

import torch
from torch import optim

from src.models.base import NeuralModel
from src.models.losses import Loss
from src.train.loaders.sampler import DataSampler
from src.train.metric_tracker import MetricTracker
from src.utils import set_seed, device


class Trainer:
    def __init__(self, model: NeuralModel, optimizer: optim.Optimizer, loss_function: Loss,
                 epochs: int, update_every: int = 1):
        self._model = model
        self._epochs = epochs
        self._optimizer = optimizer
        self._loss_function = loss_function
        self._metric_tracker = MetricTracker()

        self._step = 0
        self._update_every = update_every

    def _eval_epoch(self, epoch: int, val_loader: DataSampler, norm: float = 1.):
        self._model.eval()
        for batch in val_loader.batches():
            data, y = batch[:-1], torch.tensor(batch[-1]).to(device)
            with torch.no_grad():
                out = self._model.raw_predict(*data)
                loss = self._loss_function.get_loss(out, y)

            if loss is not None:
                self._metric_tracker.add_step(out.cpu(), y.cpu(), loss.cpu())

        self._metric_tracker.log_epoch(epoch, train=False, norm=norm)

    def _train_epoch(self, epoch: int, train_loader: DataSampler):
        self._model.train()
        for batch in train_loader.batches():
            data, y = batch[:-1], torch.tensor(batch[-1]).to(device)

            out = self._model.raw_predict(*data)
            loss = self._loss_function.get_loss(out, y)
            if loss is not None:
                (loss / self._update_every).backward()

                if (self._step + 1) % self._update_every == 0:
                    self._optimizer.step()
                    self._model.zero_grad()

                self._step += 1

                self._metric_tracker.add_step(out.cpu(), y.cpu(), loss.cpu())

        self._metric_tracker.log_epoch(epoch, train=True)

    def run_train(self, train_loader: DataSampler, val_loader: Optional[DataSampler] = None, val_norm: float = 1.):
        set_seed()
        self._step = 0

        for epoch in range(1, self._epochs + 1):
            print(f"Epoch {epoch} of {self._epochs}:")
            self._train_epoch(epoch, train_loader)

            if val_loader and len(val_loader) > 0:
                self._eval_epoch(epoch, val_loader, val_norm)
