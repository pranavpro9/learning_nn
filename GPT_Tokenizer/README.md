# GPT Tokenizer

A from-scratch implementation of Byte Pair Encoding (BPE) tokenizers, replicating the tokenization systems used in GPT-2 and GPT-4.

## Description

This project implements multiple tokenizer variants to understand how modern language models convert text into tokens. Starting from a basic BPE implementation, it progresses to regex-enhanced tokenizers and finally to a GPT-4 compatible tokenizer using OpenAI's tiktoken library.

Building tokenizers from scratch clarifies how models handle vocabulary, subword units, and special tokens - concepts that are fundamental to understanding LLM behavior.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Tokenizer Variants](#tokenizer-variants)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Tests](#tests)

## Installation

```bash
pip install torch tiktoken regex pytest
```

## Usage

### Training Tokenizers

```bash
python train.py
```

This trains three tokenizer variants on sample text and saves them to `.model` and `.vocab` files.

### Using a Trained Tokenizer

```python
from bpe.basic_bpe import BasicTokenizer
from bpe.regex import RegexTokenizer
from bpe.gpt4 import GPT4Tokenizer

# Basic tokenizer
tokenizer = BasicTokenizer()
tokenizer.load("basic.model")
tokens = tokenizer.encode("Hello, world!")
text = tokenizer.decode(tokens)

# GPT-4 tokenizer (uses pretrained cl100k_base)
gpt4 = GPT4Tokenizer()
tokens = gpt4.encode("Hello, world!")
```

## Tokenizer Variants

### BasicTokenizer
Simple BPE without regex splitting. Operates directly on UTF-8 bytes.

### RegexTokenizer
Enhanced BPE with regex patterns for better token boundaries. Handles:
- Contractions (`'s`, `'t`, `'re`)
- Words with leading spaces
- Numbers
- Special characters

### GPT4Tokenizer
Wrapper around tiktoken's `cl100k_base` encoding. Includes special token handling and is compatible with GPT-4's vocabulary.

## Project Structure

```
GPT_Tokenizer/
├── bpe/
│   ├── base.py           # Core BPE algorithm
│   ├── basic_bpe.py      # Simple BPE tokenizer
│   ├── regex.py          # Regex-enhanced tokenizer
│   └── gpt4.py           # GPT-4 tokenizer wrapper
├── test/
│   └── test_tokenizer.py # Unit tests
└── train.py              # Training script
```

## How It Works

1. **Initialization**: Start with byte-level tokens (256 base vocabulary)
2. **Pair Counting**: Count adjacent token pairs in training data
3. **Merging**: Merge the most frequent pair into a new token
4. **Iteration**: Repeat until desired vocabulary size is reached

The merge rules are saved and applied during encoding. Decoding reverses the process using a vocabulary lookup table.

## Tests

```bash
pytest test/test_tokenizer.py -v
```

Tests verify:
- Encode/decode roundtrip consistency
- Unicode handling
- Special token support
- Equivalence with tiktoken reference implementation
