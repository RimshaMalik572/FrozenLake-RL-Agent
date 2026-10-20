
# FrozenLake Reinforcement Learning Agent

## Project Overview
This project implements a Q-Learning agent using Python and Farama Gymnasium to solve the FrozenLake-v1 environment.

## Algorithm
Tabular Q-Learning with epsilon-greedy action selection.

## Training Configuration
- Environment: FrozenLake-v1
- Training episodes: 50,000
- Learning rate: 0.1
- Discount factor: 0.99
- Minimum epsilon: 0.01

## Evaluation Results
- Evaluation episodes: 100
- Successful episodes: 71
- Success rate: 71%

## Project Files
- `main.py` — training and evaluation
- `agent.py` — Q-Learning implementation
- `utils.py` — GIF recording utility
- `q_table.npy` — trained Q-table
- `training_rewards.png` — training progress graph
- `trained_agent.gif` — recorded agent episode
- `FrozenLake_RL_Assignment.ipynb` — Jupyter Notebook

## Installation
```bash
pip install -r requirements.txt
```

## Run the Project
```bash
python main.py
```

## Record the Agent
```bash
python utils.py
```
