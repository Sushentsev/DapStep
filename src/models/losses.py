from abc import ABC, abstractmethod
from math import isinf
from typing import Optional

import torch
from torch import nn

from src.utils import device


class Loss(ABC):
    @abstractmethod
    def get_loss(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CrossEntropyLoss(Loss):
    def __init__(self):
        self._loss = nn.CrossEntropyLoss()

    def get_loss(self, scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._loss(scores, target)


class RankNetLoss(Loss):
    def __init__(self, normed: bool = True):
        self._normed = normed

    def get_loss(self, scores: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
        n_classes = len(scores)

        if n_classes <= 1:
            raise ValueError("Too few scores to compute loss.")

        mask = torch.arange(n_classes).to(device) != target
        margins = (scores[target] - scores)[mask]
        loss = torch.log(torch.ones_like(margins) + torch.exp(-margins)).sum()

        if not isinf(loss.item()):
            return loss / (n_classes - 1) if self._normed else loss
