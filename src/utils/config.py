"""Configuration management for exploration strategies project."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from omegaconf import OmegaConf


@dataclass
class ExplorationConfig:
    """Configuration for exploration strategies."""
    
    # Strategy parameters
    strategy_name: str = "epsilon_greedy"
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    
    # UCB parameters
    ucb_c: float = 2.0
    
    # Thompson sampling parameters
    thompson_alpha: float = 1.0
    thompson_beta: float = 1.0
    
    # Curiosity parameters
    curiosity_intrinsic_weight: float = 0.1
    curiosity_learning_rate: float = 0.001


@dataclass
class AgentConfig:
    """Configuration for RL agents."""
    
    # Agent type
    agent_type: str = "tabular"  # "tabular" or "dqn"
    
    # Q-learning parameters
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    state_discretization: Optional[int] = None
    
    # DQN parameters
    dqn_learning_rate: float = 0.001
    dqn_hidden_size: int = 128
    dqn_num_layers: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Environment
    env_name: str = "CartPole-v1"
    seed: Optional[int] = 42
    
    # Training parameters
    num_episodes: int = 1000
    max_steps_per_episode: int = 500
    
    # Evaluation parameters
    eval_frequency: int = 100
    num_eval_episodes: int = 10
    
    # Logging and saving
    save_frequency: int = 500
    output_dir: str = "outputs"
    render: bool = False
    
    # Comparison parameters
    num_runs: int = 5
    strategies_to_compare: List[str] = field(default_factory=lambda: [
        "epsilon_greedy", "ucb", "thompson", "curiosity"
    ])


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "exploration_strategies"
    description: str = "Comparison of exploration strategies in RL"
    tags: List[str] = field(default_factory=list)


def load_config(config_path: str) -> ExperimentConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Experiment configuration
    """
    config_dict = OmegaConf.load(config_path)
    return OmegaConf.to_object(config_dict)


def save_config(config: ExperimentConfig, config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Experiment configuration
        config_path: Path to save configuration
    """
    config_dict = OmegaConf.structured(config)
    OmegaConf.save(config_dict, config_path)


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """Create default configurations for different experiments.
    
    Returns:
        Dictionary of experiment configurations
    """
    configs = {}
    
    # Basic epsilon-greedy experiment
    configs["epsilon_greedy"] = ExperimentConfig(
        exploration=ExplorationConfig(strategy_name="epsilon_greedy"),
        training=TrainingConfig(num_episodes=1000)
    )
    
    # UCB experiment
    configs["ucb"] = ExperimentConfig(
        exploration=ExplorationConfig(strategy_name="ucb", ucb_c=2.0),
        training=TrainingConfig(num_episodes=1000)
    )
    
    # Thompson sampling experiment
    configs["thompson"] = ExperimentConfig(
        exploration=ExplorationConfig(strategy_name="thompson"),
        training=TrainingConfig(num_episodes=1000)
    )
    
    # Curiosity-driven experiment
    configs["curiosity"] = ExperimentConfig(
        exploration=ExplorationConfig(strategy_name="curiosity"),
        training=TrainingConfig(num_episodes=1000)
    )
    
    # Comparison experiment
    configs["comparison"] = ExperimentConfig(
        training=TrainingConfig(
            num_episodes=1000,
            strategies_to_compare=["epsilon_greedy", "ucb", "thompson", "curiosity"],
            num_runs=5
        )
    )
    
    # DQN experiment
    configs["dqn"] = ExperimentConfig(
        agent=AgentConfig(agent_type="dqn"),
        training=TrainingConfig(num_episodes=2000)
    )
    
    return configs


# Default configuration files
DEFAULT_CONFIGS = {
    "basic": """
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
""",
    
    "comparison": """
exploration:
  strategy_name: "epsilon_greedy"

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
  num_runs: 5
  strategies_to_compare: ["epsilon_greedy", "ucb", "thompson", "curiosity"]
""",
    
    "dqn": """
exploration:
  strategy_name: "epsilon_greedy"
  epsilon: 1.0
  epsilon_decay: 0.995
  epsilon_min: 0.01

agent:
  agent_type: "dqn"
  dqn_learning_rate: 0.001
  dqn_hidden_size: 128
  dqn_num_layers: 2

training:
  env_name: "CartPole-v1"
  num_episodes: 2000
  max_steps_per_episode: 500
  eval_frequency: 100
  num_eval_episodes: 10
  seed: 42
  output_dir: "outputs"
"""
}
