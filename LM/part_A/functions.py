# /Users/panciut/Downloads/257852_marco_panciera/LM/part_A/functions.py

import torch
import torch.nn as nn

class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers=n_layers, batch_first=True)
        self.out_dropout = nn.Dropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        self.pad_token = pad_index

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
