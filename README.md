# Learning Neural Networks

An online coding journal documenting my journey of learning neural networks, tokenizers, and large language models (LLMs) through hands-on implementation.

## Description

This repository contains practical implementations of various neural network architectures and concepts, progressing from foundational machine learning principles to production-scale language models. Each project is self-contained and demonstrates specific concepts in deep learning.

The motivation behind this project was to gain a deep understanding of how modern AI systems work by building them from scratch, rather than just using pre-built libraries.

## Table of Contents

- [Projects](#projects)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [License](#license)

## Projects

| Project | Description | Key Concepts |
|---------|-------------|--------------|
| [GPT_Tokenizer](./GPT_Tokenizer) | BPE tokenizer implementations from scratch | Byte Pair Encoding, Unicode handling, GPT-4 tokenization |
| [make_more](./make_more) | Character-level name generator | MLPs, CNNs, Batch Normalization, Backpropagation |
| [Shakespear_GPT](./Shakespear_GPT) | Transformer-based text generation | Self-attention, Transformer architecture |
| [NanoGPT_GPT2_124M](./NanoGPT_GPT2_124M) | Production-ready GPT-2 training | Distributed training, Flash Attention, Mixed precision |
| [snake_game](./snake_game) | AI learns to play Snake | Reinforcement Learning, Deep Q-Networks |

## Installation

Clone the repository:

```bash
git clone https://github.com/pranavp311/learning_nn.git
cd learning_nn
```

Install dependencies (varies by project):

```bash
pip install torch numpy matplotlib
pip install tiktoken transformers  # For GPT projects
pip install pygame  # For snake_game
```

## Usage

Each subdirectory contains its own README with specific instructions. Generally:

```bash
cd <project_directory>
python <main_script>.py
```

## Technologies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- tiktoken
- Transformers (HuggingFace)
- Pygame

## License

MIT License
