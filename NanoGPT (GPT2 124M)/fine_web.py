"""
FineWeb Edu dataset (for serious pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk
Run as
$ python fine_web.py
Will save shards to local directory "edu_fineweb10B"
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard -> 100 shards (10B tokens)

# create local directory for cache if it doesnt exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# get tiktoken encoder
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc): # tokenizes a doc, returns a numpy array of unsigned int 16 tokens
    tokens = [eot] # special <|endoftext|> token delimts all documents ()
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np =  np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large, max vocab size: 65536"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

if __name__ == "__main__":
    # download dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

    # tokenize all documents and write output shards, each with shard_size tokens
    nproc = max(1, os.cpu_count()//2)
    with mp.Pool(nproc) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            # is there enough space in curr shard for new tokens
            if token_count + len(tokens) < shard_size:
                # append to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            
            else: # no space for new tokens
                # write current shard and start a new one
                split = "val" if shard_index == 0 else "train" # only first shard (index 0) is for validation rest is for training
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard -> remainder goes into the next shard
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate next shard with remainder of tokens in this doc
                all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])