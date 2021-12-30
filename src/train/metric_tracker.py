from abc import ABC, abstractmethod
from typing import Optional, Dict

import numpy as np
import torch

from src.evaluation.metrics import acc_top_k, mrr

_digits = 4


class Writer(ABC):
    @staticmethod
    @abstractmethod
    def write(metrics: Dict[str, float], epoch: int, mode: str):
        raise NotImplementedError


class StdoutWriter(Writer):
    @staticmethod
    def write(metrics: Dict[str, float], epoch: int, mode: str):
        for name, value in metrics.items():
            print(f"{mode} {name}: {round(value, _digits)}")

        print()


class MetricTracker:
    def __init__(self):
        self._epoch_losses = []
        self._y_pred = []
        self._y_true = []

        self._k_values = [1, 2, 3, 5, 10]
        self._writers = [StdoutWriter]

    def _epoch_reset(self):
        self._epoch_losses = []
        self._y_pred = []
        self._y_true = []

    def add_step(self, out: torch.Tensor, y: torch.Tensor, loss: Optional[torch.Tensor] = None):
        self._epoch_losses.append(loss.item())
        self._y_true.extend(y.tolist() if y.dim() == 1 else [y.item()])
        self._y_pred.extend(out.tolist() if out.dim() == 2 else [out.tolist()])

    def log_epoch(self, epoch: int, train: bool, norm: float = 1.):
        mode = "train" if train else "test"
        avg_loss = np.average(self._epoch_losses)
        accuracies = acc_top_k(self._y_true, self._y_pred, self._k_values)
        mrr_score = mrr(self._y_true, self._y_pred)

        if not train:
            accuracies = [accuracy * norm for accuracy in accuracies]
            mrr_score = mrr_score * norm

        metrics = {f"acc@{k}": acc for acc, k in zip(accuracies, self._k_values)}
        metrics.update({"avg loss": avg_loss, "mrr": mrr_score})

        for writer in self._writers:
            writer.write(metrics, epoch, mode)

        self._epoch_reset()
