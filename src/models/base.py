from abc import ABC, abstractmethod
from typing import List, Any

import torch
from torch import nn


class NeuralModel(ABC, nn.Module):
    @abstractmethod
    def raw_predict(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> List[Any]:
        with torch.no_grad():
            out = self.raw_predict(*args, **kwargs)
        return out.data.cpu().tolist()
