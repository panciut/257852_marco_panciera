# /LM/part_A/utils.py

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from functools import partial
import math

def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output

class Lang:
    def __init__(self, corpus, special_tokens=[]):
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    def get_vocab(self, corpus, special_tokens=[]):
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output

class PennTreeBank(data.Dataset):
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[:-1])
            self.target.append(sentence.split()[1:])

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        return {'source': src, 'target': trg}

    def mapping_seq(self, data, lang):
        res = []
        for seq in data:
            tmp = []
            for token in seq:
                if token in lang.word2id:
                    tmp.append(lang.word2id[token])
                else:
                    raise ValueError(f"OOV token found: {token}")
            res.append(tmp)
        return res

def collate_fn(data, pad_token):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = max(lengths) if lengths else 1
        padded = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            padded[i, :len(seq)] = seq
        return padded, lengths

    data.sort(key=lambda x: len(x['source']), reverse=True)
    batch = {key: [d[key] for d in data] for key in data[0]}
    source, _ = merge(batch["source"])
    target, lengths = merge(batch["target"])

    return {
        "source": source,
        "target": target,
        "number_tokens": sum(lengths)
    }

def get_dataloader(dataset, batch_size, pad_token, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate_fn, pad_token=pad_token))

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    total_loss = 0
    total_tokens = 0

    for sample in data:
        optimizer.zero_grad()
        source = sample['source'].to(next(model.parameters()).device)
        target = sample['target'].to(next(model.parameters()).device)
        output = model(source)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        total_loss += loss.item() * sample['number_tokens']
        total_tokens += sample['number_tokens']

    return total_loss / total_tokens

def eval_loop(data, criterion, model):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for sample in data:
            source = sample['source'].to(next(model.parameters()).device)
            target = sample['target'].to(next(model.parameters()).device)
            output = model(source)
            loss = criterion(output, target)
            total_loss += loss.item()
            total_tokens += sample['number_tokens']

    ppl = math.exp(total_loss / total_tokens)
    return ppl, total_loss / total_tokens

def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.RNN, nn.GRU, nn.LSTM)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)
