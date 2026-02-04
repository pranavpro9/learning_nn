# Shakespeare GPT

Transformer-based language models that generate Shakespeare-style text, implementing the GPT architecture from scratch.

## Description

This project builds GPT-like language models trained on Shakespeare's works. It includes two implementations: a simple bigram baseline and a full transformer with multi-head self-attention, demonstrating the leap in capability that attention mechanisms provide.

The goal is to understand transformer architecture by implementing every component - attention, layer normalization, positional embeddings, and the feed-forward network.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Architecture](#architecture)
- [Training](#training)

## Installation

```bash
pip install torch matplotlib
```

Place your training text in `input.txt` (Shakespeare corpus or similar).

## Usage

### Train the Bigram Model

```bash
python bigram2.py
```

### Train the Transformer GPT

```bash
python gpt.py
```

After training, both scripts generate sample text demonstrating what the model learned.

## Models

### Bigram Model (bigram2.py)
Simple baseline that predicts the next character using only the current character. Uses an embedding table where each token directly maps to logits for the next token.

### Transformer GPT (gpt.py)
Full transformer implementation with:
- Multi-head self-attention (6 heads)
- 6 transformer blocks
- Positional embeddings
- Layer normalization
- Dropout regularization
- Feed-forward networks with 4x expansion

## Architecture

```
Input Tokens
     |
Token Embeddings + Position Embeddings
     |
  [Transformer Block x 6]
     |-- Multi-Head Attention
     |-- Add & LayerNorm
     |-- Feed-Forward Network
     |-- Add & LayerNorm
     |
Layer Normalization
     |
Linear (to vocabulary)
     |
Output Logits
```

**Configuration:**
- Block size: 256 tokens
- Embedding dimension: 384
- Attention heads: 6
- Layers: 6
- Dropout: 0.2

## Training

**Hyperparameters:**
- Batch size: 64
- Learning rate: 3e-4 (AdamW)
- Iterations: 5000
- Data split: 90% train / 10% validation
- Evaluation interval: 500 steps

The training script periodically evaluates on the validation set and prints sample generations. The model typically achieves coherent Shakespeare-like text after a few thousand iterations.

**Sample Output:**
```
ROMEO:
What say'st thou? I do not know the way.

JULIET:
O, I have bought the mansion of a love,
But not possess'd it, and, though I am sold...
```
