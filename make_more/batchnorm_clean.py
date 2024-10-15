import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', "r").read().splitlines()

n_dims = 100 # number of dims space that the letter vectors are represented in
n_hidden = 200 # number of hidden layer neurons
n_letters = 27
block_size = 6 # context length
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
random.seed(42) # for reproduction
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
    
class BatchNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=1e-5):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim) # bn_gain
        self.beta = torch.zeros(dim) # bn_bias
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True) # batch mean
            xvar = x.var(0, keepdim=True) # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  
  def parameters(self):
    return []

C = torch.randn((n_letters, n_dims), generator=g)
layers = [
  Linear(n_dims * block_size, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_hidden, bias=False), BatchNorm1d(n_hidden), Tanh(),
  Linear(           n_hidden, n_letters, bias=False), BatchNorm1d(n_letters),
]

with torch.no_grad():
  # last layer: make less confident -- fix hockey stick loss appearance (useless after BN)
  layers[-1].gamma *= 0.1  

  # all other layers: apply gain -- prevent curve from squashing since tanh layer has squashing effect (useless after BN)
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 5/3

params = [C] + [p for layer in layers for p in layer.parameters()]
for p in params:
  p.requires_grad = True

for i in range(n_itr):
    # minibatch construction
    batch = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[batch], Ytr[batch]

    # forward pass
    emb = C[Xb] # embedding vectors into n_dim dimensions
    x = emb.view(batch_size, -1)
    for layer in layers:
       x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # backward pass
    '''
    for layer in layers:
    layer.out.retain_grad()
    '''
    for p in params:
       p.grad = None
    loss.backward()

    # update
    lr = decay_const**(i/1000) * 0.0651
    for p in params:
        p.data -= lr * p.grad
    
    # track stats
    '''if i % 1000 == 0: # print every 1000 steps
        print(f'{i:7d}/{n_itr:7d}: {loss.item():.4f}'
    lossi.append(loss.log10().item())'''
    with torch.no_grad():
        ud.append([((lr*p.grad).std() / p.data.std()).log10().item() for p in params]) # taking ratio of the std of ud = (Pnew - Pold) to std of Pold

# visualising saturations of the tanh layers (whether we should increase/reduce gain before batchNorm) 
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the (softmax) output layer
  if isinstance(layer, Tanh):
    t = layer.out
    print('layer %d (%10s): mean %+.2f, std %.2f, saturated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), (t.abs() > 0.97).float().mean()*100))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')

# visualising gradient distribution
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i, layer in enumerate(layers[:-1]): # note: exclude the (softmax) output layer
  if isinstance(layer, Tanh):
    t = layer.out.grad
    print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('gradient distribution')

# visualising the weights - gradient distribution
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(params):
  if p.ndim == 2: # restricts parameters to just weights for simplicity
    t = p.grad
    print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
    hy, hx = torch.histogram(t, density=True)
    plt.plot(hx[:-1].detach(), hy.detach())
    legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution')

# visualising update_value-param ratio
plt.figure(figsize=(20, 4)) # width and height of the plot
legends = []
for i,p in enumerate(params):
  if p.ndim == 2:
    plt.plot([ud[j][i] for j in range(len(ud))])
    legends.append('param %d' % i)
plt.plot([0, len(ud)], [-3, -3], 'k') # these ratios should be ~1e-3 
plt.legend(legends)

plt.show()