"""
implements gpt4-inspired tokenizer (pretrained using cl100k-base) as RegexTokenizer wrapper
"""

import tiktoken
from .regex import RegexTokenizer

def bpe(mergeable_ranks, token, max_rank):
    parts = [bytes([ch]) for ch in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks): # recover merges map from pretrained tokenizer
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]]
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank

    return merges

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
GPT4_SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100276
}

class GPT4Tokenizer(RegexTokenizer):
    def __init__(self):
        super().__init__(pattern=GPT4_SPLIT_PATTERN)
        enc = tiktoken.get_encoding("cl100k_base") # get pretrained tokenizer (called encoder in gpt4 code)
        mergeable_ranks = enc._mergeable_ranks
        
        self.merges = recover_merges(mergeable_ranks, ) # recover merges from gpt4 pretrained enc merges = (pair : idx)

        vocab = {idx : bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items(): # repopulate vocab from idx 256 onwards using merges map
            vocab[idx] = vocab[p0] + vocab[p1]
        self.vocab = vocab

        self.byte_shuffle = {i : mergeable_ranks[bytes([i])] for i in range(256)} # handle permutation
        self.byte_shuffle_inversed = {v : k for k, v in self.byte_shuffle.items()} # handle un-permutation

        self.register_special_tokens(GPT4_SPECIAL_TOKENS)


    def _encode_chunk(self, bytes_chunk):
        bytes_chunk = bytes(self.byte_shuffle[b] for b in bytes_chunk) # permute the bytes before encoding
        ids = super()._encode_chunk(bytes_chunk)
        
        return ids
    
    def decode(self, ids):
        text_bytes = b"".join(self.vocab[idx] for idx in ids)
        text_bytes = bytes(self.byte_shuffle_inversed[b] for b in text_bytes) # un-permute before decoding
        text = text_bytes.decode("utf-8", errors="replace")
        
        return text
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError("Tokenizer is pretrained")
    
    def save(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be saved.")

    def load(self, file_prefix):
        raise NotImplementedError("GPT4Tokenizer cannot be loaded.")

    def save_vocab(self, file_prefix):
        from .base import render_token
        vocab = {idx: bytes([self.byte_shuffle_inversed[idx]]) for idx in range(256)} # build vocab
        
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        
        inverted_merges = {idx: pair for pair, idx in self.merges.items()} # merge shuffled bytes
        vocab_file = file_prefix + ".vocab"
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(vocab[idx0])
                    s1 = render_token(vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

