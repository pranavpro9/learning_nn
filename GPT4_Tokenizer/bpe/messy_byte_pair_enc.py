
text = "Ｕｎｉｃｏｄｅ! 🅤🅝🅘🅒🅞🅓🅔‽ 🇺‌🇳‌🇮‌🇨‌🇴‌🇩‌🇪! 😄 The very name strikes fear and awe into the hearts of programmers worldwide. We all know we ought to “support Unicode” in our software (whatever that means—like using wchar_t for all the strings, right?). But Unicode can be abstruse, and diving into the thousand-page Unicode Standard plus its dozens of supplementary annexes, reports, and notes can be more than a little intimidating. I don’t blame programmers for still finding the whole thing mysterious, even 30 years after Unicode’s inception.".encode("utf-8")
tokens = list(map(int, text)) # explicitly convert bytes to ints

def get_stats(ids): # byte pair encoding into a hashmap recording counts
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    
    return counts

def merge(ids, pair, idx): # replace old byte pairs with new tokens
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

stats = get_stats(tokens)
#print(sorted(((v,k) for k,v in stats.items()), reverse=True))
c_vocabSize = 256 # current vocab size
t_vocabSize = 276 # target vocab size
n_merges = t_vocabSize - c_vocabSize 
ids = list(tokens) # copy to ensure original list is not changed

merges = {} # (a, b) : c
for i in range(n_merges): 
    stats = get_stats(ids)
    top_pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, top_pair, idx)
    merges[top_pair] = idx

# initialize a map of int to bytes (including the merged tokens)
vocab = {idx: bytes[idx] for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1] # byte concat

def decode(ids):
    tokens = b"".join(vocab[idx] for idx in ids) # get concat tokens in bytes from int list
    text = tokens.decode("utf-8", errors="replace") # errors set to "replace" to handle tokens predicted by LLM which do not conform to utf-8 
    return text

def encode(text):
    tokens = list(text.encode("utf-8")) # raw bytes
    while len(tokens) > 1: # ensure that we can call get_stats
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens 