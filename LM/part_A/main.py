# /Users/panciut/Downloads/257852_marco_panciera/LM/part_A/main.py

# This file is used to run your functions and print the results
# Please write your functions or classes in the functions.py

from functions import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import math
from tqdm import tqdm

if __name__ == "__main__":
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # === Load raw data ===
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, ["<pad>", "<eos>"])
    vocab_len = len(lang.word2id)

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = get_dataloader(train_dataset, batch_size=64, pad_token=lang.word2id["<pad>"], shuffle=True)
    dev_loader = get_dataloader(dev_dataset, batch_size=128, pad_token=lang.word2id["<pad>"])
    test_loader = get_dataloader(test_dataset, batch_size=128, pad_token=lang.word2id["<pad>"])

    # === Define model ===
    model = LM_LSTM(emb_size=300, hidden_size=200, output_size=vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # === Training loop ===
    best_model = None
    best_ppl = math.inf
    patience = 3
    n_epochs = 100

    for epoch in tqdm(range(1, n_epochs + 1)):
        train_loss = train_loop(train_loader, optimizer, criterion_train, model)
        dev_ppl, dev_loss = eval_loop(dev_loader, criterion_eval, model)

        print(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Dev ppl: {dev_ppl:.2f}")

        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0:
            print("Early stopping.")
            break

    best_model.to(DEVICE)
    test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print(f"Final test PPL: {test_ppl:.2f}")
