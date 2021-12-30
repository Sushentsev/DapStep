from collections import OrderedDict
from typing import Optional

import torch
from torch import nn

from src.models.ranking.similarities.base import SimModule
from src.utils import device


def _sim_input(stack_embedding: torch.Tensor, fixers_embeddings: torch.Tensor, M: torch.Tensor,
               features: Optional[torch.Tensor] = None) -> torch.Tensor:
    stack_embeddings = stack_embedding.repeat(fixers_embeddings.shape[0], 1)  # -> (fixers_num, stack_emb_size)
    sims = torch.tensor([stack_embedding @ M @ fixer_emb for fixer_emb in fixers_embeddings]).reshape(-1, 1).to(device)
    cat = torch.cat((stack_embeddings, sims, fixers_embeddings), dim=1)
    return torch.cat((cat, features), dim=1) if features is not None else cat


class MLPSimilarity(SimModule):
    def __init__(self, stack_emb_size: int, fixer_emb_size: int, features_size: int, dropout_prob: float = 0.2):
        super().__init__(stack_emb_size, fixer_emb_size, features_size, "mlp_similarity")
        self._M = torch.randn(stack_emb_size, fixer_emb_size, dtype=torch.float, requires_grad=True).to(device)
        self._dropout = nn.Dropout(dropout_prob)
        self._sim = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(stack_emb_size + fixer_emb_size + features_size + 1, 50)),
            ("relu2", nn.ReLU()),
            ("linear2", nn.Linear(50, 1))
        ]))

    def sim(self, stack_embedding: torch.Tensor, fixers_embeddings: torch.Tensor,
            features: Optional[torch.Tensor] = None) -> torch.Tensor:
        cat = _sim_input(stack_embedding, fixers_embeddings, self._M, features)
        sims = self._sim(self._dropout(cat))  # -> (fixers_size, 1)
        return sims.squeeze(1)
