from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn


class SimModule(ABC, nn.Module):
    def __init__(self, stack_emb_size: int, fixer_emb_size: int, features_size: int, name: str):
        super(SimModule, self).__init__()
        self._stack_emb_size = stack_emb_size
        self._fixer_emb_size = fixer_emb_size
        self._features_size = features_size
        self._name = name

    @abstractmethod
    def sim(self, stack_embedding: torch.Tensor, fixers_embeddings: torch.Tensor,
            features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computers similarity scores.
        :param stack_embedding: embedded stack, shape (stack_emb_size)
        :param fixers_embeddings: embedded fixers, shape (num_fixers, fixers_emb_size)
        :param features: shape (num_fixers, features_size)
        :return: similarity scores, shape (num_fixers)
        """
        raise NotImplementedError
