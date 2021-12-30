from abc import abstractmethod, ABC
from typing import Optional, List

import torch
from torch import nn

from src.models.embeddings.cnn.cnn_embedding import CNNEmbedding
from src.models.embeddings.rnn.rnn_embedding import RNNEmbedding
from src.utils import device


class NeuralEmbedding(ABC, nn.Module):
    def __init__(self, dim: int):
        super(NeuralEmbedding, self).__init__()
        self._dim = dim

    @abstractmethod
    def emb(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return self._dim


class AnnotationNeuralEmbedding(NeuralEmbedding):
    def __init__(self, emb_dim: int, hidden_size, dropout_prob: float = 0.2):
        emb = RNNEmbedding(emb_dim, hidden_size)
        super(AnnotationNeuralEmbedding, self).__init__(emb.dim)
        self._dropout = nn.Dropout(dropout_prob)
        self._emb = emb

    def emb(self, annotations: torch.Tensor, annotation_lens: torch.Tensor) -> torch.Tensor:
        """
        :param annotations: shape (batch_size, max_annotation_len, annotation_dim).
        :param annotation_lens: shape (batch_size)
        :return: shape (batch_size, annotation_emb_dim)
        """
        annotations = self._dropout(annotations)
        return self._emb(annotations, annotation_lens)


class StackNeuralEmbedding(NeuralEmbedding):
    def __init__(self, emb_module: nn.Module, vocab_size: int, emb_dim: int, dropout_prob: float = 0.2,
                 padding_idx: int = 0, pad_to_len: Optional[int] = None):
        super().__init__(emb_module.dim)
        self._word_embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx)
        self._dropout = nn.Dropout(dropout_prob)
        self._emb_module = emb_module
        self._pad_to_len = pad_to_len

    def _pad_stack(self, stacks: torch.Tensor) -> torch.Tensor:
        stacks_padded = torch.zeros(stacks.shape[0], self._pad_to_len, dtype=torch.long)
        stacks_padded[:, :stacks.shape[1]] = stacks
        return stacks_padded.to(device)

    def _pad_features(self, features: torch.Tensor) -> torch.Tensor:
        features_padded = torch.zeros(features.shape[0], self._pad_to_len, features.shape[2], dtype=torch.float)
        features_padded[:, :features.shape[1], :] = features
        return features_padded.to(device)

    def emb(self, stacks: torch.Tensor, stack_lens: torch.Tensor,
            features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param stacks: shape (batch_size, max_stack_len)
        :param stack_lens: shape (batch_size)
        :param features: shape (batch_size, max_stack_len, features_dim)
        :return: shape (batch_size)
        """
        if (self._pad_to_len is not None) and (stacks.shape[1] < self._pad_to_len):
            stacks = self._pad_stack(stacks)
            if features is not None:
                features = self._pad_features(features)

        embedded = self._word_embeddings(stacks)

        if features is not None:
            embedded = torch.cat((embedded, features), dim=-1)

        embedded = self._dropout(embedded)
        return self._emb_module(embedded, stack_lens)


class CNNStackNeuralEmbedding(StackNeuralEmbedding):
    def __init__(self, out_channels: int, kernel_heights: List[int], vocab_size: int,
                 features_dim: int = 0, embedding_dim: int = 70, dropout_prob: float = 0.2, padding_idx: int = 0):
        emb_module = CNNEmbedding(embedding_dim + features_dim, out_channels, kernel_heights)
        super(CNNStackNeuralEmbedding, self).__init__(emb_module, vocab_size, embedding_dim,
                                                      dropout_prob, padding_idx, max(kernel_heights))


class RNNStackNeuralEmbedding(StackNeuralEmbedding):
    def __init__(self, vocab_size: int, features_dim: int = 0, embedding_dim: int = 70,
                 hidden_size: int = 100, dropout_prob: float = 0.2, padding_idx: int = 0):
        emb_module = RNNEmbedding(embedding_dim + features_dim, hidden_size)
        super(RNNStackNeuralEmbedding, self).__init__(emb_module, vocab_size, embedding_dim,
                                                      dropout_prob, padding_idx)
