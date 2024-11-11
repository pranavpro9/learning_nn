"""
base tokenizer class
"""

import unicodedata

def get_stats(ids, counts=None): # byte pair encoding into a hashmap recording counts
    counts = {} if counts is None else counts
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

def replace_control_chars(s : str) -> str:
    chars = []
    for char in s:
        if unicodedata.category(char)[0] != 'C':
            chars.append(char)
        else:
            chars.append(f"\\u{ord(char):04x}") 
    
    return "".join(chars)

def render_token(tokens):
    s = tokens.decode('utf-8', errors='replace')
    s = replace_control_chars(s)
    
    return s

# base Tokenizer
class Tokenizer:
    def __init__(self):
        self.merges = {} # (idx, idx) : new token idx
        self.pattern = "" # regex pattern to force splits across certain categories of tokens/words
        self.special_tokens = {} # special token : idx
        self.vocab = self._build_vocab() # idx : bytes

    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
    
    def decode(self, ids):
        raise NotImplementedError
    
    def _build_vocab(self): # create idx : bytes mapping inclusive of merged and special tokens
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special_token, idx in self.special_tokens.items():
            vocab[idx] = special_token.encode("utf-8")
       
        return vocab
    
    def save(self, file_name): # saves file_name.vocab and file_name.model
        # model file for load() func
        # vocab file for human inspection
        model_file = file_name + ".model"
        with open(model_file, mode="w") as f:
            f.write("bpe v1\n") # writes ver
            f.write(f"{self.pattern}\n") # then writes regex pattern
            f.write(f"{len(self.special_tokens)}\n") # then writes number of special tokens
            for token, idx in self.special_tokens.items(): # then writes each special token and their respective index
                f.write(f"{token} {idx}\n")
            
            for pair, idx in self.merges: # then writes pair of merged indices and their parent index
                f.write(f"{pair} {idx}\n")

        vocab_file = file_name + ".vocab"
        inversed_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)
                if idx in inversed_merges: # for merged tokens --> finds children
                    idx0, idx1 = inversed_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else: # for original 256 tokens (leaf)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file): # loads model file
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "bpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the num of special tokens and the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merged tokens 
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()

