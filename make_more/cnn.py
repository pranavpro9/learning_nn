import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("names.txt", "r").read().splitlines()

n_dims = 100 # number of dims space that the letter vectors are represented in
n_hidden = 200 # number of hidden layer neurons
n_letters = 27
n_flattens = 3
block_size = 8 # context length
decay_const = 0.90
g = torch.Generator().manual_seed(2147483647) # for reproductability 
n_itr = 10000
batch_size = 32
ud = []
lossi = []

letters = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(letters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

def build_dataset(words):
    X, Y = [], []
    for word in words:
        context = [0] * block_size
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1]) 
Xval, Yval = build_dataset(words[n1:n2])
Xtest, Ytest = build_dataset(words[n2:])

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=g)/(fan_in**0.5)
        self.bias = torch.zeros(fan_out) if bias else None
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias != None:
            self.out += self.bias
        return self.out
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class CausalConv1D:

    def __init__(self, fan_in, fan_out, kernel_size, dilation=1): 
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = torch.nn.Conv1d(fan_in, fan_out, kernel_size, dilation=dilation)
        self.padding = (kernel_size - 1) * dilation

    def __call__(self, x):
        # Input shape: (B, T, C), Conv1d expects (B, C, T), so we need to permute
        x = x.permute(0, 2, 1) # (B, C, T)
        x = F.pad(x,(self.padding,0))
        self.conv(x)
        x = x.permute(0, 2, 1)  # Back to (B, T, C)
        return x
    
    def parameters(self):
        return self.conv.parameters()

class BatchNorm1D:

    def __init__(self, dim, eps=1e-5, momentum=1e-5):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters trained with backprop
        self.bngain = torch.ones(dim)
        self.bnbias = torch.zeros(dim)
        # parameters trained with exponential moving ave 
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        if self.training:
            if x.ndim == 2:
                dim = 0
            # ensure that mean is being taken across both 0th and 1st dims instead of just the 0th dim
            elif x.ndim == 3:
                dim = (0,1)
            xmean = x.mean(dim, keepdim=True) # batch mean
            xvar = x.var(dim, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean)/torch.sqrt(xvar + self.eps) # BN
        self.out = self.bngain * xhat + self.bnbias

        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out

    def parameters(self):
        return [self.bngain, self.bnbias]
    
class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
  
    def parameters(self):
        return []

class Embedding:

    def __init__(self, n_embeddings, n_letters):
        self.weight = torch.randn((n_embeddings, n_letters))
    
    def __call__(self, x):
        self.out = self.weight[x]
        return self.out
    
    def parameters(self):
        return [self.weight]
    
class FlattenConsecutive:

    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        B, T, C = x.shape
        x = x.reshape(B, T//self.n, C * self.n) # takes the 1st index dim (T) and splits it into n groups to then concat into 2nd index dim (C) --> same as torch.concat(x.view(:,::2,:), x.view(:,1::2,:), dim=2)
        if x.shape[1] == 1:
            x = x.squeeze(1)
        self.out = x
        return self.out
    
    def parameters(self):
        return []
    
class Sequential:
    
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def training_mode(self, bool):
        for layer in self.layers:
            layer.training = bool
    

model = Sequential([
            Embedding(n_letters, n_dims), 
            FlattenConsecutive(2), CausalConv1D(n_dims * 2, n_hidden, kernel_size=3, dilation=1), BatchNorm1D(n_hidden), Tanh(),
            FlattenConsecutive(2), CausalConv1D(n_dims * 2, n_hidden, kernel_size=3, dilation=2), BatchNorm1D(n_hidden), Tanh(),
            FlattenConsecutive(2), CausalConv1D(n_dims * 2, n_hidden, kernel_size=3, dilation=4), BatchNorm1D(n_hidden), Tanh(),
            Linear(n_hidden, n_letters),
        ])

with torch.no_grad():
  # last layer: make less confident -- fix hockey stick loss appearance (useless after BN)
  model.layers[-1].weight *= 0.1  

  # all other layers: apply gain -- prevent curve from squashing since tanh layer has squashing effect (useless after BN)
  for layer in model.layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

params = model.parameters()
for p in params:
    p.requires_grad = True

for i in range(n_itr):
    # minibatch construction 
    batch = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[batch], Ytr[batch]
    
    # forward pass
    x = Xb
    x = model(x)
    loss = F.cross_entropy(x, Yb)

    # backwards pass 
    for layer in model.layers: # remove after debug
        layer.out.retain_grad()
    for p in params:
        p.grad = None
    loss.backward()

    # update
    lr = decay_const**(i/10000) * 0.0651 
    for p in params:
        p.data -= lr * p.grad

    # track stats 
    if i % 1000 == 0: # print every _ iterations
        print(f'{i:7d}/{n_itr:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())

model.training_mode(False)

@torch.no_grad() # decorator disables gradient tracking
def split_loss(split):
    x,y = {
    'train': (Xtr, Ytr),
    'val': (Xval, Yval),
    'test': (Xtest, Ytest),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())


split_loss('train')
split_loss('val')

'''
# print shape of x after each layer --> visualise whats happening on each flatten layer (tree-like str)
for layer in model.layers:
    print(layer.__class__.__name__, ":", layer.out.shape)
'''

plt.plot(torch.tensor(lossi).view(-1, 100).mean(1))
plt.show()

'''
# sample from the model 
for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        logits = model(torch.tensor(context))
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, 1).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break
'''