import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

block_size = 8 
batch_size = 32
n_itrs = 10000
eval_itrs = 200
eval_interval = 300
lr = 1.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
n_vocab = len(chars)

# simple tokenizer --> can use more complex ones to reduce integer sequence (tradeoff between n_vocab size and size of int seq)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

# data split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:] # ensure that network isnt overfitting 

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    batch_ix = torch.randint(0, len(data) - block_size, (batch_size, ))
    x = torch.stack([data[ix: ix + block_size] for ix in batch_ix])
    y = torch.stack([data[ix + 1: ix + block_size + 1] for ix in batch_ix])
    x,y = x.to(device), y.to(device)
    return x, y

# loss averaging (across minibatches)
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # set model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_itrs)
        for k in range(eval_itrs):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads of logits for next token in table
        self.token_embedding_table = nn.Embedding(n_vocab, n_vocab)
    
    def __call__(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # B (batch - batch size), T (time -- no. time steps), C (channels - n_embeddings)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # F.cross_entropy requires C (channels) to be second dimension
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, loss = self(idx) # turns (B, T) idx into (B, C) logits
            logits = logits[:,-1,:] # taking only last timestep --> making it (B, C)
            
            # apply softmax to logits to obtain probs
            probs = torch.softmax(logits, -1) # stays (B, C) -- probablilities of each of the 65 char

            # sampling from probs to obtain each token
            next_ix = torch.multinomial(probs, 1) # (B, 1)

            # append each token
            idx = torch.cat((idx, next_ix), dim=1) # (B, T+1)
        return idx
            
# initialize model
model = BigramLanguageModel()
model = model.to(device)

# create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-1)

for itr in range(n_itrs):
    
    if itr % eval_interval == 0:
        losses = estimate_loss()
        print(f"{itr} iteration: Training loss:{losses['train']} Val loss:{losses['val']}")

    xb, yb = get_batch('train') # sample a batch

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
 

# sample from model 
idx_in = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(model.generate(idx_in, max_tokens=500)[0].tolist()))

