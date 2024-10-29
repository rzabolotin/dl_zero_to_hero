import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def tokenize(char):
    return char_to_index.get(char, 0)


def untokenize(index):
    return index_to_char.get(index, " ")


def get_batch(split, batch_size = 4):
    data = val_data if split == "valid" else train_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    X = torch.stack([data[i:i+block_size] for i in idx])
    Y = torch.stack([data[i+1:i+block_size+1] for i in idx])
    return X,Y


@torch.no_grad()
def evaluate_model(model, neval=20):
    model.eval()
    scores = {}
    for split in ['train', 'valid']:
        loss = 0
        for i in range(neval):
            X, Y = get_batch(split, batch_size=32)
            _, loss_i = model(X, Y)
            loss += loss_i.item()
        scores[split] = loss / neval
    model.train()
    return scores


class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x, target=None):
        logits = self.embedding(x)
        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]  # Use only logtis from last token
            probs = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, new_token), dim=1)
        return idx

    def generate_text(self, max_tokens=100):
        prompt = torch.zeros([1, 1], dtype=torch.long)
        return "".join([untokenize(x) for x in self.generate(prompt, max_tokens).tolist()[0]])


print("Starting ...")
print("")
print("Reading book ...")

with open("../book.txt", "r") as f_in:
    book = f_in.read()
    book = book[1681:]  # remove special info

print(f"Book has {len(book)} characters")
print(f"Book has {len(set(book))} unique characters")
print("Example text:")
print(book[1000:2000])
# Tokenizer

vocab = sorted(list(set("".join(book))), key=lambda v: "\t" if v == "." else v)
vocab_size = len(vocab)

char_to_index = {char: index for index, char in enumerate(vocab)}
index_to_char = {index: char for char, index in char_to_index.items()}

# Prepare data for training

data = torch.tensor([tokenize(x) for x in book], dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8 

model = BigramModel(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("")
print("")
print("Training ...")
print("")
print("")

for i in range(10000):
    X,Y = get_batch('train', batch_size=32)
    logits, loss = model(X, Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        scores = evaluate_model(model)
        print(f"Loss train: {scores['train']:.4f}, valid {scores['valid']:.4f}")

print("")
print("")
print("Generating text ...")
print("")
print("")

print(model.generate_text(1000))
print("")
