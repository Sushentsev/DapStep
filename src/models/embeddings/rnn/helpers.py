from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

NEG_INF = -1e32


def _seq_mask(seq_lens: torch.Tensor) -> torch.Tensor:
    """
    :param seq_lens: shape (batch_size)
    :return: mask, shape (batch_size, max_seq_len)
    """

    max_seq_len = torch.max(seq_lens).item()
    idx = torch.arange(max_seq_len).to(seq_lens).repeat(seq_lens.shape[0], 1)
    mask = torch.gt(seq_lens.unsqueeze(1), idx).to(seq_lens)  # -> (batch_size, max_seq_len)

    return mask


def _mask_softmax(scores: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    :param scores: not normalizer attention scores, shape (batch_size, max_seq_len)
    :param mask: shape (batch_size, max_seq_len)
    :return: shape (batch_size, max_seq_len)
    """

    if mask is not None:
        mask_norm = (1 - mask) * NEG_INF
        scores = scores + mask_norm

    result = F.softmax(scores, dim=1)
    return result


class SoftAttentionConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, last_hidden: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        """
        :param outputs: shape (batch_size, max_sent_len, output_dim)
        :param last_hidden: shape (batch_size, hidden_dim)
        :param seq_lens: shape (batch_size)
        :return: shape (batch_size, 2 * output_dim)
        """

        assert outputs.shape[2] == last_hidden.shape[1]
        batch_size = outputs.shape[0]
        last_hidden = last_hidden.unsqueeze(-1)  # -> (batch_size, hidden_dim, 1)
        attention_scores = outputs.bmm(last_hidden).squeeze(-1)  # -> (batch_size, max_sent_len)
        mask = _seq_mask(seq_lens)  # -> (batch_size, max_sent_len)
        attention_scores = _mask_softmax(attention_scores, mask)  # -> (batch_size, max_sent_len)
        attention = torch.sum(attention_scores.unsqueeze(-1) * outputs, dim=1)  # -> (batch_size, output_dim)
        last_outputs = outputs[torch.arange(batch_size), seq_lens - 1]  # -> (batch_size, output_dim)
        result = torch.cat((last_outputs, attention), 1)  # -> (batch_size, 2 * output_dim)
        return result


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self._lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, emb_seqs: torch.Tensor, seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param emb_seqs: shape (batch_size, max_sent_length, embedding_dim)
        :param seq_lens: shape (batch_size)
        :return: tuple (y, hidden)
        """
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        emb_seqs_sorted = torch.index_select(emb_seqs, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        emb_seqs_packed = pack_padded_sequence(emb_seqs_sorted, seq_lens_sort.cpu(), batch_first=True)
        y_packed, (hidden, _) = self._lstm(emb_seqs_packed)

        # hidden.shape = (2 (forward and backward), batch_size, hidden_size)

        y_sort, y_length = pad_packed_sequence(y_packed, batch_first=True)
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)
        hidden = torch.index_select(hidden, dim=1, index=idx_unsort)  # -> (2, batch_size, hidden_size)
        return y, hidden
