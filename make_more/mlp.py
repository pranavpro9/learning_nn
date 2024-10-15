import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open('names.txt', "r").read().splitlines()

n_dims = 200 # number of dims space that the letter vectors are represented in
n_hidden = 300 # number of hidden layer neurons
n_letters = 27
block_size = 6 # context length
decay_const = 0.85
g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn(n_letters, n_dims,              generator=g)
W1 = torch.randn(block_size * n_dims, n_hidden, generator=g) * 0.0001
b1 = torch.randn(n_hidden,                      generator=g) * 0.1
W2 = torch.randn(n_hidden, n_letters,           generator=g) * 0.01
b2 = torch.randn(n_letters,                     generator=g) * 0
param = [C, W1, W2, b1, b2]
n_itr = 10000

letters = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(letters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# building the dataset (generate inputs and target outputs) 
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + '.':
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

'''lr_exp = torch.linspace(-1,-3, n_itr)
lrs = 10**lr_exp

lr_itrd = []
steps_itrd = []
loss_itrd = []'''

for i in range(1):
    # randomizing and plucking a minibatch for training
    batch = torch.randint(0, Xtr.shape[0], (32,))

    # forward pass
    emb = C[Xtr[batch]] # same as F.one_hot(Xb, num_classes=27) @ C --> can be visualized as bringing datapoints (X) into a 2d space using weights of C
    emb_joined = torch.tanh(emb.view(-1, block_size * n_dims)) # same as torch.cat(torch.unbind(emb, 1), 1) --> joins 1st and 2nd dimensions so that emb can undergo mult/add w weights and biases
    h = emb_joined @ W1 + b1
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Ytr[batch])
    #print(loss.item())
    
    # backward pass
    for p in param:
        p.grad = None
    loss.backward()

    # update
    lr = decay_const**(i/1000) * 0.0457
    for p in param:
        p.data -= lr * p.grad

    # to estimate a decent lr using pyplot
    '''steps_itrd.append(i)
    lr_itrd.append(lrs[i])
    loss_itrd.append(loss.item())'''

#dev/val
emb = C[Xval]
emb_joined = emb.view(-1, block_size * n_dims)
h = torch.tanh(emb_joined @ W1 + b1)
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Yval) 
print(loss.item())

# sample from the model
'''g = torch.Generator().manual_seed(2147483647 + 10)

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
    
    print(''.join(itos[i] for i in out))'''

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