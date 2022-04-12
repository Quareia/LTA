import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy
from utils import *


class BiLSTM(nn.Module):
    """ BiLSTM Model."""
    def __init__(self, input_size, hidden_size, num_layers, bi):
        """Bi-LSTM Encoder

        Args:
            input_size: (int) vocab word2vec dim
            hidden_size: (int) hidden size in Bi-LSTM
            num_layers: (int) num_layers in Bi-LSTM
            bi: (boolean) Bi-direction
        """
        super(BiLSTM, self).__init__()

        # init
        # # Bi-LSTM
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bi = bi

        # models
        self.rnn = nn.LSTM(self.input_size,
                           self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=True,
                           bidirectional=self.bi)

    def forward(self, *input):
        self.rnn.flatten_parameters()

        x_emb, x_len, return_type = input  # (batch_size, max_len, word2vec_dim) (batch_size, )

        # BiLSTM
        total_length = x_len.max()

        x_packed = nn.utils.rnn.pack_padded_sequence(x_emb, x_len.cpu(), batch_first=True, enforce_sorted=False)
        out_lstm, hidden = self.rnn(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_lstm, batch_first=True, total_length=total_length)

        # vector represent
        if return_type == 'mean_pooling':
            out = out.sum(dim=1).div(x_len.float().unsqueeze(-1))  # (batch_size, num_directions * hidden_size)
        elif return_type == 'all_return':  # (batch_size, max_len, num_directions * hidden_size)
            pass
        return out


