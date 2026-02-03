# Exploration Strategies in Reinforcement Learning

A research-ready implementation of various exploration strategies in reinforcement learning, featuring comprehensive evaluation, statistical analysis, and interactive demos.

## Overview

This project implements and compares multiple exploration strategies for reinforcement learning:

- **Epsilon-Greedy**: Traditional random exploration with decaying probability
- **Upper Confidence Bound (UCB)**: Optimistic exploration based on confidence intervals
- **Thompson Sampling**: Bayesian exploration using posterior sampling
- **Curiosity-Driven**: Intrinsic motivation based on prediction error

## Features

- Modern PyTorch 2.x implementation with proper device handling
- Comprehensive evaluation framework with statistical analysis
- Interactive Streamlit demo for strategy comparison
- Support for both tabular Q-learning and Deep Q-Networks (DQN)
- Reproducible experiments with deterministic seeding
- Professional project structure with proper documentation

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- MPS (optional, for Apple Silicon)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Exploration-Strategies-in-Reinforcement-Learning.git
cd Exploration-Strategies-in-Reinforcement-Learning
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

### Basic Training

Train a single exploration strategy:

```bash
python scripts/train.py --strategy epsilon_greedy --num_episodes 1000
```

### Strategy Comparison

Compare multiple exploration strategies:

```bash
python scripts/train.py --compare --num_runs 5
```

### Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo/app.py
```

## Usage

### Command Line Interface

The main training script supports various options:

```bash
python scripts/train.py \
    --strategy epsilon_greedy \
    --agent_type tabular \
    --num_episodes 1000 \
    --seed 42 \
    --output_dir outputs
```

**Available options:**
- `--strategy`: Exploration strategy (`epsilon_greedy`, `ucb`, `thompson`, `curiosity`)
- `--agent_type`: Agent type (`tabular`, `dqn`)
- `--num_episodes`: Number of training episodes
- `--seed`: Random seed for reproducibility
- `--output_dir`: Output directory for results
- `--render`: Render environment during training
- `--compare`: Compare multiple strategies
- `--num_runs`: Number of runs for comparison

### Programmatic Usage

```python
from src.algorithms.q_learning import QLearningAgent
from src.algorithms.exploration_strategies import create_exploration_strategy
from src.evaluation.trainer import Trainer, TrainingConfig

# Create agent with exploration strategy
agent = QLearningAgent(
    action_space_size=2,
    exploration_strategy="epsilon_greedy",
    seed=42
)

# Configure training
config = TrainingConfig(
    env_name="CartPole-v1",
    num_episodes=1000,
    seed=42
)

# Train agent
trainer = Trainer(agent, config, "outputs")
stats = trainer.train()

# Plot results
trainer.plot_training_curves()
```

## Exploration Strategies

### Epsilon-Greedy

The classic exploration strategy that balances exploitation and exploration:

```python
strategy = create_exploration_strategy(
    "epsilon_greedy",
    epsilon=1.0,
    epsilon_decay=0.995,
    epsilon_min=0.01
)
```

**Parameters:**
- `epsilon`: Initial exploration rate
- `epsilon_decay`: Decay rate for epsilon
- `epsilon_min`: Minimum epsilon value

### Upper Confidence Bound (UCB)

Optimistic exploration that considers both estimated value and uncertainty:

```python
strategy = create_exploration_strategy(
    "ucb",
    c=2.0  # Exploration parameter
)
```

**Parameters:**
- `c`: Exploration parameter (higher = more exploration)

### Thompson Sampling

Bayesian exploration using posterior sampling:

```python
strategy = create_exploration_strategy(
    "thompson",
    alpha=1.0,  # Prior alpha parameter
    beta=1.0    # Prior beta parameter
)
```

**Parameters:**
- `alpha`: Prior alpha parameter for Beta distribution
- `beta`: Prior beta parameter for Beta distribution

### Curiosity-Driven Exploration

Intrinsic motivation based on prediction error:

```python
strategy = create_exploration_strategy(
    "curiosity",
    intrinsic_weight=0.1,
    learning_rate=0.001
)
```

**Parameters:**
- `intrinsic_weight`: Weight for intrinsic reward
- `learning_rate`: Learning rate for prediction network

## Evaluation Metrics

The framework provides comprehensive evaluation metrics:

- **Learning Performance**: Average return, success rate, time-to-threshold
- **Sample Efficiency**: Steps to reach performance threshold
- **Stability**: Reward variance, catastrophic resets
- **Statistical Significance**: Confidence intervals, t-tests
- **Exploration Analysis**: State visitation, action distribution

## Project Structure

```
exploration-strategies-rl/
├── src/
│   ├── algorithms/
│   │   ├── exploration_strategies.py  # Core exploration strategies
│   │   └── q_learning.py             # Q-learning agents
│   ├── evaluation/
│   │   └── trainer.py                # Training and evaluation framework
│   └── utils/
│       ├── config.py                # Configuration management
│       └── utils.py                  # Utility functions
├── configs/                          # Configuration files
├── scripts/
│   └── train.py                      # Main training script
├── demo/
│   └── app.py                        # Streamlit demo
├── tests/                            # Unit tests
├── notebooks/                        # Jupyter notebooks
├── assets/                           # Generated plots and results
├── requirements.txt                  # Dependencies
├── pyproject.toml                    # Project configuration
└── README.md                         # This file
```

## Configuration

The project uses OmegaConf for configuration management. Example configuration:

```yaml
exploration:
  strategy_name: "epsilon_greedy"
  epsilon: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01

agent:
  agent_type: "tabular"
  learning_rate: 0.1
  discount_factor: 0.99

training:
  env_name: "CartPole-v1"
  num_episodes: 1000
  max_steps_per_episode: 500
  eval_frequency: 100
  num_eval_episodes: 10
  seed: 42
  output_dir: "outputs"
```

## Results and Visualization

### Training Curves

The framework automatically generates training curves showing:
- Episode rewards over time
- Evaluation performance
- Episode lengths
- Exploration rate decay

### Comparison Analysis

Statistical comparison includes:
- Box plots for reward distributions
- Confidence intervals
- Significance tests
- Performance rankings

### Interactive Demo

The Streamlit demo provides:
- Real-time strategy comparison
- Configurable parameters
- Interactive visualizations
- Downloadable results

## Development

### Code Quality

The project uses modern Python development practices:

- Type hints throughout
- Comprehensive docstrings
- Black code formatting
- Ruff linting
- Pre-commit hooks

### Testing

Run tests with:

```bash
pytest tests/
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Run pre-commit hooks
5. Submit a pull request

## Safety and Limitations

**IMPORTANT DISCLAIMER**: This project is designed for research and educational purposes only. It is NOT intended for production control of real-world systems.

### Limitations

- **Environment Compatibility**: Tested primarily on CartPole-v1
- **Scalability**: Tabular methods limited by state space size
- **Hyperparameter Sensitivity**: Performance may vary with different parameter settings
- **Computational Requirements**: DQN requires significant computational resources

### Safety Considerations

- **Deterministic Seeding**: All experiments use fixed seeds for reproducibility
- **Resource Limits**: Built-in wall-clock time limits prevent runaway training
- **Error Handling**: Comprehensive error handling and graceful degradation
- **Documentation**: Clear documentation of limitations and assumptions

## Citation

If you use this project in your research, please cite:

```bibtex
@software{exploration_strategies_rl,
  title={Exploration Strategies in Reinforcement Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Exploration-Strategies-in-Reinforcement-Learning}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI Gym/Gymnasium for the RL environments
- PyTorch team for the deep learning framework
- Streamlit for the interactive demo framework
- The RL research community for foundational algorithms

## Contact

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Contact the maintainers
- Join the discussion forum

---

**Note**: This project is maintained for research and educational purposes. For production applications, please ensure proper testing, validation, and safety measures are in place.
# Exploration-Strategies-in-Reinforcement-Learning
