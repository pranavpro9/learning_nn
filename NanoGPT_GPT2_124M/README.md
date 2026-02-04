# NanoGPT GPT-2 124M

A production-ready GPT-2 implementation with distributed training, mixed precision, and benchmark evaluation capabilities.

## Description

This project implements GPT-2 from scratch with all the optimizations needed for training at scale. It can load pretrained weights from HuggingFace, train on large datasets using multiple GPUs, and evaluate on the HellaSwag benchmark.

The implementation demonstrates modern deep learning engineering practices: Flash Attention, gradient accumulation, cosine learning rate scheduling, and distributed data parallel training.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Configurations](#model-configurations)
- [Training Pipeline](#training-pipeline)
- [Evaluation](#evaluation)
- [Data Preparation](#data-preparation)

## Installation

```bash
pip install torch transformers tiktoken datasets numpy tqdm
```

For multi-GPU training, ensure CUDA and NCCL are properly configured.

## Usage

### Data Preparation

Download and tokenize the FineWeb-Edu dataset:

```bash
python fine_web.py
```

This creates tokenized shards in numpy format for efficient loading.

### Training

**Single GPU:**
```bash
python train_gpt2.py
```

**Multi-GPU (4 GPUs):**
```bash
torchrun --nproc_per_node=4 train_gpt2.py
```

### Evaluation

Evaluate on HellaSwag benchmark:

```bash
python hellaswag.py -m gpt2 -d cuda
```

### Loading Pretrained Weights

```python
from train_gpt2 import GPT

model = GPT.from_pretrained('gpt2')  # 124M
model = GPT.from_pretrained('gpt2-medium')  # 350M
model = GPT.from_pretrained('gpt2-large')  # 774M
model = GPT.from_pretrained('gpt2-xl')  # 1558M
```

## Model Configurations

| Model | Layers | Heads | Embedding Dim | Parameters |
|-------|--------|-------|---------------|------------|
| GPT-2 | 12 | 12 | 768 | 124M |
| GPT-2 Medium | 24 | 16 | 1024 | 350M |
| GPT-2 Large | 36 | 20 | 1280 | 774M |
| GPT-2 XL | 48 | 25 | 1600 | 1558M |

## Training Pipeline

**Architecture Features:**
- Flash Attention via `scaled_dot_product_attention`
- GELU activation in feed-forward blocks
- Pre-normalization (LayerNorm before attention/FFN)
- Weight tying between embeddings and output layer

**Training Features:**
- Distributed Data Parallel (DDP) for multi-GPU
- Gradient accumulation for large effective batch sizes
- BF16 mixed precision (optional)
- Cosine learning rate schedule with warmup
- Gradient clipping (norm = 1.0)
- Periodic checkpointing

**Default Hyperparameters:**
- Total batch size: 524,288 tokens
- Micro batch: 2
- Sequence length: 1024
- Max steps: 19,073 (~10B tokens)
- Warmup: 715 steps
- Learning rate: 6e-4 (max), 6e-5 (min)
- Weight decay: 0.1
- Optimizer: AdamW with fused kernels

## Evaluation

The HellaSwag evaluation (`hellaswag.py`) tests commonsense reasoning:

1. Download HellaSwag validation set
2. For each example, compute loss on 4 possible completions
3. Predict the completion with lowest loss
4. Report accuracy

## Data Preparation

`fine_web.py` handles dataset preparation:

1. Downloads FineWeb-Edu from HuggingFace (10B tokens subset)
2. Tokenizes using GPT-2 tokenizer (tiktoken)
3. Splits into shards (~100M tokens each)
4. Saves as numpy arrays for fast loading

The data loader supports multi-process reading for efficient GPU utilization during training.
