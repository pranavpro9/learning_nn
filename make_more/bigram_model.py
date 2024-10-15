import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

# opens file for reading, then reads in everything into a massive string, then splits the massive string
words = open('names.txt', 'r').read().splitlines() 

g = torch.Generator().manual_seed(2147483647)

letters = sorted(list(set(''.join(words))))
vector_size = len(letters) + 1

stoi = {s:i+1 for i,s in enumerate(letters)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


# bigram training (no neural net)
N = torch.zeros((vector_size,vector_size), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1

P = (N+1).float()
P /= P.sum(1, keepdim=True)



# generate 5 names using bigram training (no neural net)
for i in range(0):
    out = [] 
    ix = 0 
    while True:
        p = P[ix]
        ix = torch.multinomial(p, 1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))


# creating list of input and target output bigrams
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)


xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()

# single layer neural network with 27 neurons of random weights 
W = torch.randn((27,27), generator=g, requires_grad=True)

for k in range(100):
    # forward pass + one-hot encoding of inputs
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = xenc @ W 
    counts = logits.exp()
    probs = counts/counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.1 * (W**2).mean()
    print(loss.item())

    # backward pass
    W.grad = None
    loss.backward()

    # update
    W.data += -60 * W.grad 


# plot count-map of all possible bigrams
'''plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()'''

