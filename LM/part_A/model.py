# /LM/part_A/model.py

import torch
import torch.nn as nn

class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, 
                 n_layers=1, use_lstm=False, emb_dropout=0.0, out_dropout=0.0):
        super(LM_RNN, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
        self.rnn_type = "LSTM" if use_lstm else "RNN"
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True) if use_lstm else nn.RNN(emb_size, hidden_size, n_layers, batch_first=True)
        self.out_dropout = nn.Dropout(out_dropout) if out_dropout > 0 else nn.Identity()
        self.output = nn.Linear(hidden_size, output_size)
        self.pad_token = pad_index

    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))
        rnn_out, _ = self.rnn(emb)
        rnn_out = self.out_dropout(rnn_out)
        output = self.output(rnn_out).permute(0, 2, 1)
        return output
