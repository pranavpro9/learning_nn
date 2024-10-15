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

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item() # check exactly the same
  app = torch.allclose(dt, t.grad) # check approx the same
  maxdiff = (dt - t.grad).abs().max().item() # max diff in value
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

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

for p in param:
    p.requires_grad = True
 
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
    bnmeani = (1/n) * hprebn.sum(0, keepdim=True)
    bndiff = hprebn - bnmeani
    bndiffsq = bndiff**2
    bnvar = (1/(n-1)) * bndiffsq.sum(0, keepdim=True)
    bnvar_inv = (bnvar + 1e-5) ** -0.5
    bnraw = bndiff * bnvar_inv
    hpreact = bngain * bnraw + bnbias
    # non-linearity layer
    h = torch.tanh(hpreact)
    # linear layer 2 
    logits = h @ W2 + b2
    # cross entropy loss
    logit_maxes = logits.max(1, keepdim=True).values
    norm_logits = logits - logit_maxes # prevent too large numbers in counts
    counts = norm_logits.exp()
    counts_sums = counts.sum(1, keepdim=True)
    counts_s_inv = counts_sums ** -1 # if 1/counts is used instead, backprop is not bit exact
    probs = counts * counts_s_inv
    logprobs = probs.log()
    loss = -logprobs[range(n), Yb].mean()

    # PyTorch backward pass
    for p in param:
        p.grad = None
    for t in [logprobs, probs, counts, counts_sums, counts_s_inv,
        norm_logits, logit_maxes, logits, h, hpreact, bnraw,
        bnvar_inv, bnvar, bndiffsq, bndiff, hprebn, bnmeani,
        embcat, emb]:
        t.retain_grad()
    loss.backward()

    # manual backward pass
    dlogprobs = torch.zeros_like(logprobs)
    dlogprobs[range(n), Yb] = -1.0/n # since loss = (-1/n) * logprob

    dprobs = 1.0/probs * dlogprobs

    dcounts_s_inv = (counts * dprobs).sum(1, keepdim=True)

    dcounts = counts_s_inv * dprobs
    dcounts_sums = -(counts_sums**-2) * dcounts_s_inv
    dcounts += torch.ones_like(counts) * dcounts_sums # a11 + a12 + a13 = b1, hence ∂b1/∂(a11), ∂b1/∂(a12), ∂b1/∂(a13) are all 1 (also note that derivitive is being added)

    dnorm_logits = counts * dcounts # since counts is norm_logits.exp() anyways
    dlogit_maxes = -dnorm_logits.sum(1, keepdim=True)
    dlogits = dnorm_logits.detach().clone()
    dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes # in x1, x2, x3 --> xmax = x1 --> ∂xmax/∂x1 is 1 while ∂xmax/∂x2 and ∂xmax/∂x3 is 0
    dh = dlogits @ W2.T
    dW2 = h.T @ dlogits
    db2 = dlogits.sum(0)
    dhpreact = (1-h**2) * dh # y = tanh(x), dy/dx = 1 - tanh^2(x) = 1 - y^2
    dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    dbnraw = bngain * dhpreact
    dbnbias = dhpreact.sum(0, keepdim=True)
    dbndiff = bnvar_inv * dbnraw
    dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
    dbnvar = -0.5 * dbnvar_inv * (bnvar + 1e-5) ** -1.5 
    dbndiffsq = torch.ones_like(bndiffsq) * (1.0/(n-1)) * dbnvar
    dbndiff += 2 * bndiff * dbndiffsq
    dhprebn = dbndiff.clone()
    dbnmeani = (-dbndiff).sum(0, keepdim=True)
    dhprebn += torch.ones_like(hprebn) * (1/n) * dbnmeani
    dembcat = dhprebn @ W1.T
    dW1 = embcat.T @ dhprebn
    db1 = dhprebn.sum(0)
    demb = dembcat.view(emb.shape)

    X_T = F.one_hot(Xb, num_classes=27).flatten(start_dim=0,end_dim=1).T.float()
    demb_f = demb.flatten(start_dim=0,end_dim=1)
    dC = X_T @ demb_f

    '''
    dC = torch.zeros_like(C)
    for k in range(Xb.shape[0]):
    for j in range(Xb.shape[1]):
    ix = Xb[k,j]
    dC[ix] += demb[k,j]
    '''

    # update
    lr = decay_const**(i/1000) * -0.0651
    for p in param:
        p.data += lr * p.grad

cmp('logprobs', dlogprobs, logprobs) 
cmp('probs', dprobs, probs)
cmp('counts_s_inv', dcounts_s_inv, counts_s_inv)
cmp('counts_sums', dcounts_sums, counts_sums)
cmp('counts', dcounts, counts)
cmp('norm_logits', dnorm_logits, norm_logits)
cmp('logit_maxes', dlogit_maxes, logit_maxes)
cmp('logits', dlogits, logits)
cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)
cmp('hpreact', dhpreact, hpreact)
cmp('bngain', dbngain, bngain)
cmp('bnbias', dbnbias, bnbias)
cmp('bnraw', dbnraw, bnraw)
cmp('bnvar_inv', dbnvar_inv, bnvar_inv)
cmp('bnvar', dbnvar, bnvar)
cmp('bndiffsq', dbndiffsq, bndiffsq)
cmp('bndiff', dbndiff, bndiff)
cmp('bnmeani', dbnmeani, bnmeani)
cmp('hprebn', dhprebn, hprebn)
cmp('embcat', dembcat, embcat)
cmp('W1', dW1, W1)
cmp('b1', db1, b1)
cmp('emb', demb, emb)
cmp('C', dC, C)

