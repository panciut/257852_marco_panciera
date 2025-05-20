# /LM/part_B/main.py

from model import LM_LSTM_1B
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math
import os
import json
import datetime
import time

# === Configurable Flags ===
USE_WEIGHT_TYING = False
USE_VARIATIONAL_DROPOUT = False
USE_NTASGD = False

# === Hyperparameters ===
EMB_SIZE = 300
HID_SIZE = 300
LR = 1.0
CLIP = 5
BATCH_SIZE = 64
N_EPOCHS = 100
PATIENCE = 3
DROPOUT = 0
PAD_TOKEN = "<pad>"
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# === Timestamped experiment naming ===
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"part1B_drop{DROPOUT}_lr{LR}_wt{USE_WEIGHT_TYING}_vd{USE_VARIATIONAL_DROPOUT}_ntasgd{USE_NTASGD}"
MODEL_DIR = f"models/{EXPERIMENT_NAME}"
LOG_DIR = f"logs/{EXPERIMENT_NAME}"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, f"model_{TIMESTAMP}.pt")
METRICS_JSON_PATH = os.path.join(LOG_DIR, f"metrics_{TIMESTAMP}.json")
HPARAMS_PATH = os.path.join(LOG_DIR, f"hparams_{TIMESTAMP}.json")

# === Save config ===
hparams = {
    "EMB_SIZE": EMB_SIZE,
    "HID_SIZE": HID_SIZE,
    "DROPOUT": DROPOUT,
    "LR": LR,
    "USE_WEIGHT_TYING": USE_WEIGHT_TYING,
    "USE_VARIATIONAL_DROPOUT": USE_VARIATIONAL_DROPOUT,
    "USE_NTASGD": USE_NTASGD,
    "BATCH_SIZE": BATCH_SIZE,
    "DEVICE": DEVICE,
}
with open(HPARAMS_PATH, 'w') as f:
    json.dump(hparams, f, indent=4)

# === Load Data ===
train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

lang = Lang(train_raw, [PAD_TOKEN, "<eos>"])
vocab_size = len(lang.word2id)

train_data = PennTreeBank(train_raw, lang)
dev_data = PennTreeBank(dev_raw, lang)
test_data = PennTreeBank(test_raw, lang)

train_loader = get_dataloader(train_data, batch_size=BATCH_SIZE, pad_token=lang.word2id[PAD_TOKEN], shuffle=True)
dev_loader = get_dataloader(dev_data, batch_size=BATCH_SIZE, pad_token=lang.word2id[PAD_TOKEN])
test_loader = get_dataloader(test_data, batch_size=BATCH_SIZE, pad_token=lang.word2id[PAD_TOKEN])

# === Model ===
model = LM_LSTM_1B(
    emb_size=EMB_SIZE,
    hidden_size=HID_SIZE,
    output_size=vocab_size,
    pad_index=lang.word2id[PAD_TOKEN],
    dropout=DROPOUT if USE_VARIATIONAL_DROPOUT else 0.0,
    weight_tying=USE_WEIGHT_TYING
).to(DEVICE)
model.apply(init_weights)

# === Optimizer: SGD → NTASGD
optimizer = optim.SGD(model.parameters(), lr=LR)
triggered = False
nonmono_count = 0
t_trigger = 3

# === Losses
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id[PAD_TOKEN])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id[PAD_TOKEN], reduction='sum')

# === Training
best_model = None
best_ppl = math.inf
patience = PATIENCE
metrics_history = []

start_time = time.time()

for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=CLIP)
    dev_ppl, dev_loss = eval_loop(dev_loader, criterion_eval, model)

    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Dev loss: {dev_loss:.4f} | Dev ppl: {dev_ppl:.2f}")

    metrics_history.append({
        "epoch": epoch,
        "train_loss": round(train_loss, 4),
        "dev_loss": round(dev_loss, 4),
        "dev_ppl": round(dev_ppl, 2)
    })

    if dev_ppl < best_ppl:
        best_ppl = dev_ppl
        best_model = copy.deepcopy(model).to('cpu')
        patience = PATIENCE
        nonmono_count = 0
    else:
        nonmono_count += 1
        patience -= 1

    if USE_NTASGD and nonmono_count >= t_trigger and not triggered:
        print(f"Switching to NT-ASGD at epoch {epoch}")
        optimizer = optim.ASGD(model.parameters(), lr=LR)
        triggered = True

    if patience <= 0:
        print("Early stopping.")
        break

# === Test
best_model.to(DEVICE)
test_ppl, test_loss = eval_loop(test_loader, criterion_eval, best_model)

# === Save everything
torch.save(best_model.state_dict(), MODEL_PATH)
with open(METRICS_JSON_PATH, 'w') as f:
    json.dump(metrics_history, f, indent=4)

print("\n===== FINAL RESULTS =====")
print(f"Best Dev PPL: {best_ppl:.2f}")
print(f"Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")
print(f"Model saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_JSON_PATH}")
print(f"HParams saved to: {HPARAMS_PATH}")
print(f"Training time: {(time.time() - start_time) / 60:.2f} minutes")
