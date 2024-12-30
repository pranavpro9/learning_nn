"""
Tokenizer training on long wikipedia articles
"""

import os
import time
from bpe import BasicTokenizer, RegexTokenizer, GPT4Tokenizer

# open some text and train a vocab of 512 tokens
text = open("test/Tartan.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
# construct the Tokenizer object and kick off verbose training
for TokenizerType, name in zip([BasicTokenizer, RegexTokenizer, GPT4Tokenizer], ["basic", "regex", "gpt4"]):
    tokenizer = TokenizerType()
    file_prefix = os.path.join("models", name)
    
    if TokenizerType in [BasicTokenizer, RegexTokenizer]:
        tokenizer.train(text, 512, verbose=True) 
        # writes name.model, and name.vocab
        tokenizer.save(file_prefix)
    else: # for gpt4 tokenizer
        tokenizer.save_vocab(file_prefix)
    t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")