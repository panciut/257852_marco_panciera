# /LM/part_A/main.py

from model import *
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import math
from tqdm import tqdm
import os
import logging

# === Hyperparameters ===
EMB_SIZE = 300
HID_SIZE = 200
LR = 0.0001
CLIP = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL = 128
PAD_TOKEN = "<pad>"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# === Setup Directories ===
MODEL_DIR = "models"
LOG_DIR = "logs"
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_rnn.pt")
LOG_PATH = os.path.join(LOG_DIR, "training_output.txt")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Logger Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger()

if __name__ == "__main__":
    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    lang = Lang(train_raw, [PAD_TOKEN, "<eos>"])
    vocab_len = len(lang.word2id)

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    train_loader = get_dataloader(train_dataset, batch_size=BATCH_SIZE_TRAIN, pad_token=lang.word2id[PAD_TOKEN], shuffle=True)
    dev_loader = get_dataloader(dev_dataset, batch_size=BATCH_SIZE_EVAL, pad_token=lang.word2id[PAD_TOKEN])
    test_loader = get_dataloader(test_dataset, batch_size=BATCH_SIZE_EVAL, pad_token=lang.word2id[PAD_TOKEN])

    model = LM_RNN(emb_size=EMB_SIZE, hidden_size=HID_SIZE, output_size=vocab_len, pad_index=lang.word2id[PAD_TOKEN]).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id[PAD_TOKEN])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id[PAD_TOKEN], reduction='sum')

    best_model = None
    best_ppl = math.inf
    patience = 3
    n_epochs = 100

    for epoch in tqdm(range(1, n_epochs + 1)):
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=CLIP)
        dev_ppl, dev_loss = eval_loop(dev_loader, criterion_eval, model)

        log.info(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Dev ppl: {dev_ppl:.2f}")

        if dev_ppl < best_ppl:
            best_ppl = dev_ppl
            best_model = copy.deepcopy(model).to('cpu')
            patience = 3
        else:
            patience -= 1

        if patience <= 0:
            log.info("Early stopping.")
            break

    best_model.to(DEVICE)
    test_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    log.info(f"Final test PPL: {test_ppl:.2f}")

    torch.save(best_model.state_dict(), MODEL_PATH)
    log.info(f"Model saved to: {MODEL_PATH}")
