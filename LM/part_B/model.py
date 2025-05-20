# /LM/part_B/model.py

import torch
import torch.nn as nn

class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        # x: (batch, time, features)
        mask = x.new_empty(1, x.size(1), x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        mask = mask.expand_as(x)
        return x * mask

class LM_LSTM_1B(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 n_layers=1, dropout=0.0, weight_tying=True):
        super().__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.locked_dropout = LockedDropout(dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.output_dropout = LockedDropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

        if weight_tying:
            if emb_size != hidden_size:
                raise ValueError("Weight tying requires emb_size == hidden_size")
            self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.locked_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.output_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
