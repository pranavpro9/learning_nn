"""
basic Byte Pair Encoder (BPE) similar to GPT2 tokenizer
https://github.com/openai/gpt-2/blob/master/src/encoder.py
    WITH REGEX splitting pattern
    WITH (optional) special token handling
"""

from .base import Tokenizer, get_stats, merge
import regex as re

GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {} # special token : idx
        self.special_tokens_inversed = {} # idx : special token
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        n_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text) # regex pattern forces splits across certain categories of tokens/words

        ids = list(chunk.encode("utf-8") for chunk in text_chunks)

        # BPE
        merges = {}
        vocab = {idx : bytes([idx]) for idx in range(256)}
        for i in range(n_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats) # get counts of each pair (within each chunk)
            
            max_pair = max(stats, key=stats.get)
            new_idx = 256 + i 
            ids = [merge(chunk_ids, max_pair, new_idx) for chunk_ids in ids] # get new ids after replacing merged pairs

            merges[max_pair] = new_idx # append pair and new parent idx to merges map
            vocab[new_idx] = vocab[max_pair[0]] + vocab[max_pair[1]] # append new parent index to vocab

            if verbose: # prints
                print(f"merge {i+1}/{n_merges}: {max_pair} -> {new_idx} ({vocab[new_idx]}) had {stats[max_pair]} occurrences")
        
        self.merges = merges
        self.vocab = vocab
    
    def register_special_tokens(self, special_tokens): # special token : idx
        self.special_tokens = special_tokens
        self.special_tokens_inversed = {v: k for k, v in special_tokens.items()}
    
    def _encode_chunk(self, bytes_chunk):
        ids = list(bytes_chunk)
        while len(ids) >= 2:
            stats = get_stats(ids)
            min_pair = min(stats, key=lambda p : self.merges.get(p, float("inf"))) # returns min parent idx of merged pair in stats
            
            if min_pair not in self.merges: # catch case where if no more merges available -> min_pair will be inf
                break

            idx = self.merges[min_pair]
            ids = merge(ids, min_pair, idx)
        
        return ids

    def encode_basic(self, text): # takes a string of text and returns ids (without considering special tokens)
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            bytes_chunk = chunk.encode("utf-8")
            ids_chunk = self._encode_chunk(bytes_chunk)
            ids.extend(ids_chunk)
        
        return ids
        
    def encode(self, text, allowed_special="none_raise"): # takes a string of text and returns ids (considering special tokens)
        special = None
        if allowed_special == "all": # handle all special tokens
            special = self.special_tokens
        
        elif allowed_special == "none": # handle none of the special tokens
            special = {}
        
        elif allowed_special == "none_raise": # no special tokens in text
            special = {}
            assert all(token not in text for token in self.special_tokens)
        
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        
        else: # no preference indicated
            raise ValueError(f"allowed_special={allowed_special} is invalid. \n usage: allowed_special=[ all | none | none_raise | (set_of_special_chars)]")
        
        if not special: # if special = None use the normal encoding instead
            return self.encode_basic(text)
        
        special_pattern = "(" + "|".join(re.escape(token) for token in special) + ")" # init a capture group containing all the special characters in special
        special_chunks = re.split(special_pattern, text)
        ids = []
        for chunk in special_chunks:
            if chunk in special: # if token special
                ids.append(special[chunk]) # append idx of special token
            else: # normal token seq
                ids.extend(self.encode_basic(chunk))

        return ids
        
    
    def decode(self, ids): # takes a list of ids and returns string text
        bytes_list = []
        for idx in ids:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
            
            elif idx in self.special_tokens_inversed:
                bytes_list.append(self.special_tokens_inversed[idx].encode("utf-8")) # use encode to get bytes since special_token inversed[idx] give the special token itself
            
            else:
                raise ValueError(f"Token of index {idx} is invalid")
        
        text_bytes = b"".join(bytes_list)
        text = text_bytes.decode("utf-8", errors="replace")
        
        return text
