import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', 'r').read().splitlines()

n_dims = 100 # number of dims space that the letter vectors are represented in
n_hidden = 200 # number of hidden layer neurons
n_letters = 27
block_size = 6 # context length
decay_const = 0.90
g = torch.Generator().manual_seed(2147483647) # for reproducibility 
n_itr = 10000
batch_size = 32
n = batch_size 
momentum = 1e-5
bn_mean_running = torch.zeros((1,n_hidden))
bn_var_running = torch.ones((1, n_hidden))

# non-standard init is used because when doing manual backprop, init w zeros can mask incorrect backprop implementations
C = torch.randn((n_letters, n_dims),              generator=g)
W1 = torch.randn((n_dims * block_size, n_hidden), generator=g) * (5/3 * (block_size * n_dims)**(-0.5))
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # still useless because of batchnorm
W2 = torch.randn((n_hidden, n_letters),           generator=g) * 0.1
b2 = torch.randn(n_letters,                       generator=g) * 0.1
bngain = torch.randn((1, n_hidden)) * 0.1 + 1
bnbias = torch.randn((1, n_hidden)) * 0.1

param = [C, W1, W2, b1, b2, bngain, bnbias]

letters = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(letters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# utility function to use later when comparing manual gradients to PyTorch gradients
'''
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item() # check exactly the same
  app = torch.allclose(dt, t.grad) # check approx the same
  maxdiff = (dt - t.grad).abs().max().item() # max diff in value
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
'''

# building the dataset (generate inputs and target outputs) 
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.' + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

# extracting training dataset, dev/val dataset --> for hyperparams, test dataset
random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1]) 
Xval, Yval = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

@torch.no_grad() # decorator disables gradient tracking
def split_loss(split):
    x,y = {
    'train': (Xtr, Ytr),
    'val': (Xval, Yval),
    'test': (Xtest, Ytest),
    }[split]
    emb = C[x] 
    embcat = emb.view(-1, block_size * n_dims)
    hpreact = embcat @ W1
    hpreact = bngain * (hpreact - bn_mean_running)/(bn_var_running + 1e-5) + bnbias 
    h = torch.tanh(hpreact) 
    logits = h @ W2 + b2 
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

for p in param:
    p.requires_grad = True
 
with torch.no_grad():
    for i in range(n_itr):
        
        # construct minibatch
        batch = torch.randint(0, Xtr.shape[0], (batch_size,),)
        Xb, Yb = Xtr[batch], Ytr[batch]

        # Forward Pass
        # embedding minibatch into n_dims
        emb = C[Xb]
        embcat = emb.view(batch_size, -1)
        # linear layer 1
        hprebn = embcat @ W1 + b1
        # batchnorm layer
        bnmean = hprebn.mean(0, keepdim=True)
        bnvar = hprebn.var(0, keepdim=True, unbiased=True)
        bnvar_inv = (bnvar + 1e-5)**-0.5
        bnraw = (hprebn - bnmean) * bnvar_inv
        hpreact = bngain * bnraw + bnbias

        bn_mean_running = (1 - momentum) * bn_mean_running + (momentum) * bnmean
        bn_var_running = (1 - momentum) * bn_var_running + (momentum) * bnvar
        # non-linearity layer
        h = torch.tanh(hpreact)
        # linear layer 2 
        logits = h @ W2 + b2
        # cross entropy loss
        loss = F.cross_entropy(logits, Yb)

        # mathematically determined backward pass
            # Since loss = -log[(e**y)/sum(e**i)] where i = logits
            # if i != y, dloss/di = probi = (e**i)/(sum(e**i))
            # if i = y, dloss/di = probi - 1 = (e**i)/(sum(e**i)) - 1

        dlogits = F.softmax(logits, 1)
        dlogits[range(n), Yb] -= 1
        dlogits /= n

        dh = dlogits @ W2.T
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(0)

        dhpreact = (1-h**2) * dh # y = tanh(x), dy/dx = 1 - tanh^2(x) = 1 - y^2

        dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
        dbnbias = dhpreact.sum(0, keepdim=True)
        dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))

        dembcat = dhprebn @ W1.T
        dW1 = embcat.T @ dhprebn
        db1 = dhprebn.sum(0)

        demb = dembcat.view(emb.shape)
        X_T = F.one_hot(Xb, num_classes=27).flatten(start_dim=0,end_dim=1).T.float()
        demb_f = demb.flatten(start_dim=0,end_dim=1)
        dC = X_T @ demb_f

        grads = [dC, dW1, dW2, db1, db2, dbngain, dbnbias]

        # update
        lr = decay_const**(i/1000) * -0.0651
        for p, grad in zip(param, grads):
            p.data += lr * grad

split_loss('train')
split_loss('val')

# sample from the model
'''
g = torch.Generator().manual_seed(2147483647 + 10)


for _ in range(20):

    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      # ------------
      # forward pass:
      # Embedding
      emb = C[torch.tensor([context])] # (1,block_size,d)      
      embcat = emb.view(emb.shape[0], -1) # concat into (N, block_size * n_embd)
      hpreact = embcat @ W1 + b1
      hpreact = bngain * (hpreact - bnmean) * (bnvar + 1e-5)**-0.5 + bnbias
      h = torch.tanh(hpreact) # (N, n_hidden)
      logits = h @ W2 + b2 # (N, vocab_size)
      # ------------
      # Sample
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
'''