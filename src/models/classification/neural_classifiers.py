from typing import List

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from src.models.base import NeuralModel
from src.models.convert import get_stack_values
from src.models.embeddings.embeddings import StackNeuralEmbedding, CNNStackNeuralEmbedding, \
    RNNStackNeuralEmbedding
from src.preprocess.token import Token
from src.utils import device


class NeuralClassifier(NeuralModel):
    def __init__(self, emb_module: StackNeuralEmbedding, classifier: nn.Module):
        super().__init__()
        self._emb_module = emb_module
        self._classifier = classifier

    def raw_predict(self, stacks: List[List[Token[int]]]) -> torch.Tensor:
        stack_lens = torch.tensor([len(stack) for stack in stacks], dtype=torch.long).to(device)
        stacks = pad_sequence([torch.tensor(get_stack_values(stack), dtype=torch.long)
                               for stack in stacks], batch_first=True, padding_value=0).to(device)
        embeddings = self._emb_module.emb(stacks, stack_lens)
        logits = self._classifier(embeddings)
        return logits


class CNNNeuralClassifier(NeuralClassifier):
    def __init__(self, num_classes: int, vocab_size: int, out_channels: int, kernel_heights: List[int],
                 embedding_dim: int = 70, dropout_prob: float = 0.2, padding_idx: int = 0):
        emb_module = CNNStackNeuralEmbedding(out_channels, kernel_heights, vocab_size, embedding_dim=embedding_dim,
                                             dropout_prob=dropout_prob, padding_idx=padding_idx)
        classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(emb_module.dim, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )
        super().__init__(emb_module, classifier)


class RNNNeuralClassifier(NeuralClassifier):
    def __init__(self, num_classes: int, vocab_size: int, embedding_dim: int = 70, hidden_size: int = 100,
                 dropout_prob: float = 0.2, padding_idx: int = 0):
        emb_module = RNNStackNeuralEmbedding(vocab_size=vocab_size, embedding_dim=embedding_dim,
                                             hidden_size=hidden_size, dropout_prob=dropout_prob,
                                             padding_idx=padding_idx)
        classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(emb_module.dim, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )

        super().__init__(emb_module, classifier)
