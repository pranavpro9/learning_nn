# Snake Game AI

A classic Snake game with both human-playable and AI-controlled versions, where the AI learns to play using Deep Q-Learning.

## Description

This project implements the Snake game twice: once for human players and once for an AI agent that learns through reinforcement learning. The AI uses a neural network to approximate Q-values and learns optimal gameplay through trial and error.

Watching the AI improve from random movements to strategic food-seeking behavior demonstrates how reinforcement learning works in practice.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How the AI Works](#how-the-ai-works)
- [Training](#training)
- [Hyperparameters](#hyperparameters)

## Installation

```bash
pip install torch pygame matplotlib numpy
```

Note: The game requires a display. For headless environments, consider using a virtual display.

## Usage

### Human Player

```bash
cd snake_game_human
python app.py
```

Controls: Arrow keys to change direction.

### AI Training

```bash
cd snake_game_AI
python agent.py
```

The AI will start training immediately, displaying a real-time plot of scores.

## Project Structure

```
snake_game/
├── snake_game_human/
│   └── app.py              # Human-playable game
└── snake_game_AI/
    ├── agent.py            # Q-learning agent
    ├── model.py            # Neural network and trainer
    ├── AI_snake_game.py    # AI-compatible game environment
    └── helper.py           # Training visualization
```

## How the AI Works

### State Representation
The agent observes 11 boolean features:
- **Danger** (3): Straight, right, or left of current direction
- **Direction** (4): Current movement direction (up/down/left/right)
- **Food location** (4): Relative position of food (up/down/left/right)

### Actions
Three possible actions encoded as `[straight, right, left]`:
- `[1, 0, 0]` - Continue straight
- `[0, 1, 0]` - Turn right
- `[0, 0, 1]` - Turn left

### Rewards
- `+10` - Eating food
- `-10` - Death (wall collision or self-collision)
- `0` - Other moves

### Neural Network
```
Input (11) -> Linear (256) -> ReLU -> Linear (3) -> Output
```

### Learning Algorithm
Deep Q-Learning with experience replay:
1. Observe state, choose action (epsilon-greedy)
2. Execute action, observe reward and next state
3. Store experience in replay memory
4. Sample batch from memory
5. Update Q-values using Bellman equation

## Training

The agent trains continuously, with performance plotted in real-time:
- Blue line: Score per game
- Orange line: Mean score

Training typically shows improvement within 50-100 games, with the AI achieving consistent high scores after a few hundred games.

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 0.001 | Adam optimizer step size |
| Gamma | 0.9 | Discount factor for future rewards |
| Epsilon start | 80 | Initial exploration rate |
| Epsilon decay | Linear | Decays to 0 over 80 games |
| Batch size | 1000 | Experience replay batch |
| Memory size | 100,000 | Maximum stored experiences |
| Hidden neurons | 256 | Single hidden layer size |

The model saves its best weights to `model/model.pth` when achieving new high scores.
