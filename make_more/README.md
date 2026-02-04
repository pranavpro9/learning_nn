# Make More

A character-level name generator exploring neural network architectures from bigram models to convolutional networks, with deep dives into batch normalization and manual backpropagation.

## Description

This project trains neural networks to generate new names by learning patterns from a dataset of existing names. It demonstrates the progression of neural network complexity and provides hands-on implementations of core deep learning concepts like backpropagation and batch normalization.

The implementations prioritize educational clarity over performance, making it easier to understand what each component does and why.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Scripts](#scripts)
- [Key Concepts](#key-concepts)
- [Data](#data)

## Installation

```bash
pip install torch matplotlib
```

## Usage

Each script is standalone. Run any of them to train and generate names:

```bash
python mlp.py
python cnn.py
python bigram_model.py
```

## Scripts

### bigram_model.py
Baseline statistical model using bigram probabilities. Predicts the next character based only on the current character.

### mlp.py
Multi-layer perceptron implementation:
- Character embeddings (configurable dimension)
- Hidden layers with tanh activation
- Cross-entropy loss
- Train/validation/test splits (80/10/10)

### cnn.py
Convolutional approach to name generation:
- Embedding layer for characters
- 1D causal convolutions
- Batch normalization
- Custom layer classes demonstrating the architecture

### batchnorm.py / batchnorm_clean.py
Deep exploration of batch normalization:
- Manual implementation of running mean/variance
- Momentum-based exponential moving average
- Visualization of activation distributions
- Gradient flow analysis

### manual_backprop.py
Complete manual backpropagation without autograd:
- Hand-coded gradient calculations
- Chain rule applied step by step
- Comparison with PyTorch gradients
- Batch norm backward pass

### math_backprop.py
Mathematical approach to backpropagation:
- Gradient computation using analytical formulas
- Memory-efficient implementation
- No intermediate variable retention

## Key Concepts

**Embeddings**: Converting discrete characters to continuous vectors that capture semantic relationships.

**Batch Normalization**: Normalizing layer inputs to stabilize training and allow higher learning rates.

**Backpropagation**: Computing gradients through the chain rule, implemented both manually and with autograd.

**Context Windows**: Using multiple previous characters to predict the next one.

## Data

The `names.txt` file contains a dataset of names used for training. The model learns character patterns and generates new names that follow similar patterns.

Example generated names after training:
```
emma
olivia
ava
sophia
```
