import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', "r").read().splitlines()

n_dims = 100 # number of dims space that the letter vectors are represented in
n_hidden = 200 # number of hidden layer neurons
n_letters = 27
block_size = 6 # context length
decay_const = 0.80
g = torch.Generator().manual_seed(2147483647)

C = torch.randn(n_letters, n_dims,              generator=g)
W1 = torch.randn(block_size * n_dims, n_hidden, generator=g) * (5/3 * (block_size * n_dims)**(-0.5)) # multiply by gain/root(fan_in) to ensure std dev of gaussian distb remains relatively constant --> fixes saturated tanh (in case BN is not used)
b1 = torch.randn(n_hidden,                      generator=g) * 0.1 # (gets obliterated by BN so b1 has no impact)
W2 = torch.randn(n_hidden, n_letters,           generator=g) * 0.1 # fixes hockey stick loss graph
b2 = torch.randn(n_letters,                     generator=g) * 0.1

bn_gain = torch.ones((1, n_hidden))
bn_bias = torch.zeros((1,n_hidden))
bn_mean_running = torch.zeros((1,n_hidden))
bn_std_running = torch.ones((1, n_hidden))

param = [C, W1, W2, b2, bn_gain, bn_bias]
n_itr = 10000
momentum = 1e-5

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
    hpreact = bn_gain * (hpreact - bn_mean_running)/(bn_std_running + 1e-5) + bn_bias # batch normalization - better way to ensure std dev relatively const + also acts as a regularizer (preventing overfitting) as mean and std change with selected batch
    h = torch.tanh(hpreact) 
    logits = h @ W2 + b2 
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())

for p in param:
    p.requires_grad = True

#lr_exp = torch.linspace(-2,-5, n_itr)
#lrs = 10**lr_exp

lr_itrd = []
steps_itrd = []
loss_itrd = []

for i in range(n_itr):
    # randomizing and plucking a minibatch for training
    batch = torch.randint(0, Xtr.shape[0], (32,),)

    # forward pass
    emb = C[Xtr[batch]] # same as F.one_hot(X, num_classes=27) @ C --> can be visualized as bringing datapoints (X) into a 2d space using weights of C
    embcat = emb.view(-1, block_size * n_dims) # same as torch.cat(torch.unbind(emb, 1), 1) --> joins 1st and 2nd dimensions so that emb can undergo mult/add w weights and biases
    # weight layer (linear layer)
    hpreact = embcat @ W1 # b1 is now useless with batchnorm 
    
    # normalization layer
    bn_meani = hpreact.mean(0, keepdim=True)
    bn_stdi = hpreact.std(0, keepdim=True)
    hpreact = bn_gain * (hpreact - bn_meani/(bn_stdi + 1e-5)) + bn_bias # batch normalization - better way to ensure std dev relatively const (compared to indiv multp the weights) + also acts as a regularizer (preventing overfitting) as mean and std change with selected batch (OVERALL CAN STILL BE QUITE BUGGY -- TRY GROUP/LINEAR NORMALIZATION)
  
    with torch.no_grad():
      bn_mean_running = (1 - momentum) * bn_mean_running + (momentum) * bn_meani
      bn_std_running = (1 - momentum) * bn_std_running + (momentum) * bn_stdi

    # non-linearity
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[batch])
    #print(loss.item())
    
    # backward pass
    for p in param:
        p.grad = None
    loss.backward()

    # update
    lr = decay_const**(i/1000) * 0.0651
    for p in param:
        p.data -= lr * p.grad

    # to estimate a decent lr using pyplot
    '''steps_itrd.append(i)
    lr_itrd.append(lrs[i])
    loss_itrd.append(loss.item())'''

split_loss('train')
split_loss('val')

# sample from the model
'''
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ W1 + b1)
      logits = h @ W2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))
'''
# visualize the embeddings of each letter as a 2d vector
'''plt.figure(figsize=(8,8))
plt.scatter(C[:,0].data, C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
plt.grid(which='minor')
plt.show()'''


# plot to estimate ideal learning rate
'''plt.plot(steps_itrd, loss_itrd)
plt.show()'''
