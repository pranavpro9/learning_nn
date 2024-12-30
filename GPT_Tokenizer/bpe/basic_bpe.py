"""
basic Byte Pair Encoder (BPE) similar to GPT2 tokenizer
https://github.com/openai/gpt-2/blob/master/src/encoder.py
    without REGEX splitting pattern
    without special token handling
"""
from .base import Tokenizer, get_stats, merge

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        n_merges = vocab_size - 256
        
        # text preprocessing
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes) # list of int from 0 to 255

        # BPE
        merges = {} # idx pair : parent idx
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for i in range(n_merges):
            stats = get_stats(ids)
            max_pair = max(stats, key=stats.get)

            new_idx = 256 + i
            ids = merge(ids, max_pair, new_idx) # get new ids after replacing merged pairs
            
            merges[max_pair] = new_idx # append pair and new parent idx to merges map
            vocab[new_idx] = vocab[max_pair[0]] + vocab[max_pair[1]] # append new parent index to vocab

            if verbose: # prints
                print(f"merge {i+1}/{n_merges}: {max_pair} -> {new_idx} ({vocab[new_idx]}) had {stats[max_pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, text): # given string return ids
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf"))) # encode from lowest idx merge pair (as a parent of a merge can be in a child pair of another merge)
            
            if pair not in self.merges: # if no more merges available, pair will be inf 
                break
            
            parent_idx = self.merges[pair] # if pair in merges
            ids = merge(ids, pair, parent_idx)
        
        return ids
        
    def decode(self, ids): # return text given ids
        text_bytes = b"".join(self.vocab[idx] for idx in ids) # get bytes using vocab map
        text = text_bytes.decode("utf-8", errors="replace") # get text from bytes (replace incase token does not conform to utf-8)
        return text