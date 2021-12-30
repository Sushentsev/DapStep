import torch
from torch import nn

from src.models.embeddings.rnn.helpers import SoftAttentionConcat, BiLSTM
from src.utils import device


class RNNEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, hidden_size: int = 100):
        super(RNNEmbedding, self).__init__()
        self._hidden_size = hidden_size
        self._bi_lstm = BiLSTM(embedding_dim, hidden_size)
        self._attention_concat = SoftAttentionConcat()

    def forward(self, emb_seqs: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
        batch_size, max_sentence_length, _ = emb_seqs.shape

        y, hidden = self._bi_lstm(emb_seqs, seq_lens)
        # hidden.shape = (2 (forward and backward), batch_size, hidden_size)

        y = y.view(batch_size, max_sentence_length, 2,
                   self._hidden_size)  # -> (batch_size, max_sent_len, 2, hidden_size)
        forward_index = torch.tensor(0, dtype=torch.long).to(device)
        backward_index = torch.tensor(1, dtype=torch.long).to(device)
        y_forward, h_forward = torch.index_select(y, 2, forward_index), hidden[0]
        y_backward, h_backward = torch.index_select(y, 2, backward_index), hidden[1]
        y_forward = y_forward.squeeze(2)
        y_backward = y_backward.squeeze(2)

        # y_forward.shape = y_backward.shape = (batch_size, max_sentence_len, hidden_size)
        # h_forward.shape = h_backward.shape = batch_size, hidden_size)

        attention_forward = self._attention_concat(y_forward, h_forward, seq_lens)
        attention_backward = self._attention_concat(y_backward, h_backward, seq_lens)

        # attention_forward.shape = attention_backward.shape = (batch_size, 2 * hidden_size)

        cat = torch.cat((attention_forward, attention_backward), 1)  # -> (batch_size, 4 * hidden_size)
        return cat

    @property
    def dim(self) -> int:
        return 4 * self._hidden_size
