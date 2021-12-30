from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, out_channels: int, kernel_heights: List[int]):
        super(CNNEmbedding, self).__init__()
        self._num_kernels = len(kernel_heights)
        self._out_channels = out_channels
        self._conv_list = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=(kernel_height, embedding_dim))
            for kernel_height in kernel_heights
        ])

    def _conv_block(self, input: torch.Tensor, conv_layer: nn.Module) -> torch.Tensor:
        # input.shape = (batch_size, 1, max_sent_length, embedding_dim)
        conv_out = conv_layer(input)  # -> (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))  # -> (batch_size, out_channels, dim)
        max_out = F.max_pool1d(activation, kernel_size=activation.shape[2]).squeeze(-1)  # -> (batch_size, out_channels)
        return max_out

    def forward(self, emb_seqs: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        embedded = emb_seqs.unsqueeze(1)  # -> (batch_size, 1, max_sent_len, embedding_dim + features_dim)
        pool_list = [self._conv_block(embedded, conv_layer)
                     for conv_layer in self._conv_list]  # List of (batch_size, out_channels)
        embedded = torch.cat(pool_list, 1)  # -> (batch_size, num_kernels * out_channels)
        return embedded

    @property
    def dim(self) -> int:
        return self._num_kernels * self._out_channels
