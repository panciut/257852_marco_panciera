

from functions import *
from utils import *
from model import LM_RNN
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import math
from tqdm import tqdm
import os
import logging
import json
import datetime
import time

# === Hyperparameters ===
EMB_SIZE = 300
HID_SIZE = 200
LR = 0.0001
CLIP = 5
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_EVAL = 128
PAD_TOKEN = "<pad>"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# === Model Configurations for Part 1.A ===
USE_LSTM = False            # Use RNN (baseline)
USE_EMB_DROPOUT = False     # No embedding dropout
USE_OUT_DROPOUT = False     # No output dropout
USE_ADAMW = False           # Use SGD (baseline)


# === Experiment Setup ===
EXPERIMENT_NAME = f"lstm_baseline_lr{LR}_emb{EMB_SIZE}_hid{HID_SIZE}"
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = f"models/{EXPERIMENT_NAME}"
LOG_DIR = f"logs/{EXPERIMENT_NAME}"
MODEL_PATH = os.path.join(MODEL_DIR, f"model_{TIMESTAMP}.pt")
LOG_PATH = os.path.join(LOG_DIR, f"log_{TIMESTAMP}.txt")
METRICS_PATH = os.path.join(LOG_DIR, "metrics.csv")
HPARAMS_PATH = os.path.join(LOG_DIR, "hparams.json")

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

# === Save Hyperparameters ===
hparams = {
    "EMB_SIZE": EMB_SIZE,
    "HID_SIZE": HID_SIZE,
    "LR": LR,
    "CLIP": CLIP,
    "BATCH_SIZE_TRAIN": BATCH_SIZE_TRAIN,
    "BATCH_SIZE_EVAL": BATCH_SIZE_EVAL,
    "PAD_TOKEN": PAD_TOKEN,
    "DEVICE": DEVICE,
    "MODEL_TYPE": "LM_RNN",
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "TIMESTAMP": TIMESTAMP
}
with open(HPARAMS_PATH, 'w') as f:
    json.dump(hparams, f, indent=4)

if __name__ == "__main__":
    start_time = time.time()

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

    with open(METRICS_PATH, 'w') as metrics_file:
        metrics_file.write("epoch,train_loss,dev_loss,dev_ppl\n")

    for epoch in tqdm(range(1, n_epochs + 1)):
        train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=CLIP)
        dev_ppl, dev_loss = eval_loop(dev_loader, criterion_eval, model)

        log.info(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Dev loss: {dev_loss:.4f} | Dev ppl: {dev_ppl:.2f}")

        with open(METRICS_PATH, 'a') as metrics_file:
            metrics_file.write(f"{epoch},{train_loss:.4f},{dev_loss:.4f},{dev_ppl:.2f}\n")

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
    test_ppl, test_loss = eval_loop(test_loader, criterion_eval, best_model)

    total_time = (time.time() - start_time) / 60

    log.info("\n===== SUMMARY =====")
    log.info(f"Best Dev PPL: {best_ppl:.2f}")
    log.info(f"Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")
    log.info(f"Optimizer: SGD")
    log.info(f"Training time: {total_time:.2f} minutes")
    log.info(f"Model saved to: {MODEL_PATH}")
    log.info("===================\n")

    torch.save(best_model.state_dict(), MODEL_PATH)
