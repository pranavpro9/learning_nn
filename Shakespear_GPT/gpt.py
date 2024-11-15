import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as pyplot

block_size = 256
batch_size = 64
n_itrs = 5000
n_embd = 384
n_head = 6
n_layer = 6
dropout_p = 0.2 # chance of a neuron dropping out
eval_itrs = 200
eval_interval = 500
lr = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
n_vocab = len(chars)

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]

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

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.values = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.values(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,C) @ (B,C,T) then divide by sqrt C to keep wei roughly gaussian --> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)

        out = wei @ v # (B,T,T) @ (B,T,C)

        return out
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # 4 * n_embd as seen in the paper ( innerDim = 4*outerDim)
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection layer
            nn.Dropout(dropout_p)
            
        )
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa_heads = MultiHead(num_heads, head_size)
        self.feedfwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x)) # deviating from paper to apply ln before sa_heads and feedfwd instead of after
        x = x + self.feedfwd(self.ln2(x)) # this is done on a per token level (independent of one another)--> letting each token "think" on the data collected thru self attention
        
        return x
    
class GPTLangModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads off logits for next token in table
        self.token_embd_table = nn.Embedding(n_vocab, n_embd) 
        self.position_embd_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(Block(n_embd, 4), 
                                    Block(n_embd, 4),
                                    Block(n_embd, 4),
                                    Block(n_embd, 4), # multiple layers of self attention and feedfwd 
                                    nn.LayerNorm(n_embd),) 
        self.lm_head = nn.Linear(n_embd, n_vocab)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_embd = self.token_embd_table(idx) # B (batch - batch size), T (time -- no. time steps), C (channels = n_embeddings here)
        pos_embd = self.position_embd_table(torch.arange(T, device=device)) # (T,C)
        x = token_embd + pos_embd # (B, T, C) + (T, C) broadcasted across B
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
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
            idx_cond = idx[:,-block_size:] # take last block_size number of elements
            logits, loss = self(idx_cond) # turns (B, T) idx into (B, C) logits
            logits = logits[:,-1,:] # taking only last timestep --> making it (B, C)
            
            # apply softmax to logits to obtain probs
            probs = torch.softmax(logits, -1) # stays (B, C) -- probablilities of each of the 65 char

            # sampling from probs to obtain each token
            next_ix = torch.multinomial(probs, 1) # (B, 1)

            # append each token
            idx = torch.cat((idx, next_ix), dim=1) # (B, T+1)
        return idx
            
# initialize model
model = GPTLangModel()
model = model.to(device)

# create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

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
