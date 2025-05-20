# /LM/part_B/main.py

from model import LM_LSTM_1B
from utils import *
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import math

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

# === Optimizer: SGD → NTASGD switch
optimizer = optim.SGD(model.parameters(), lr=LR)
use_avg = False
triggered = False
avg_trigger_epoch = None
avg_params = None
t_trigger = 3  # Trigger after this many non-improving epochs

# === Loss
criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id[PAD_TOKEN])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id[PAD_TOKEN], reduction='sum')

# === Training
best_ppl = math.inf
patience = PATIENCE
nonmono_count = 0
best_model = None

for epoch in range(1, N_EPOCHS + 1):
    train_loss = train_loop(train_loader, optimizer, criterion_train, model, clip=CLIP)
    dev_ppl, dev_loss = eval_loop(dev_loader, criterion_eval, model)

    print(f"[Epoch {epoch}] Train loss: {train_loss:.4f} | Dev loss: {dev_loss:.4f} | Dev ppl: {dev_ppl:.2f}")

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
        avg_trigger_epoch = epoch

    if patience <= 0:
        print("Early stopping.")
        break

# === Test
best_model.to(DEVICE)
test_ppl, test_loss = eval_loop(test_loader, criterion_eval, best_model)
print("\n===== FINAL RESULTS =====")
print(f"Best Dev PPL: {best_ppl:.2f}")
print(f"Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")
