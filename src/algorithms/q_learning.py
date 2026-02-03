"""Modern Q-learning agent with multiple exploration strategies."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import Env

from .exploration_strategies import ExplorationStrategy, create_exploration_strategy


class QLearningAgent:
    """Modern Q-learning agent with configurable exploration strategies.
    
    This agent implements tabular Q-learning with support for various
    exploration strategies and proper state discretization for continuous
    environments.
    """

    def __init__(
        self,
        action_space_size: int,
        exploration_strategy: Union[str, ExplorationStrategy] = "epsilon_greedy",
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        state_discretization: Optional[int] = None,
        seed: Optional[int] = None,
        **strategy_kwargs
    ):
        """Initialize Q-learning agent.
        
        Args:
            action_space_size: Number of possible actions
            exploration_strategy: Strategy name or instance
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            state_discretization: Number of bins for state discretization
            seed: Random seed for reproducibility
            **strategy_kwargs: Additional arguments for exploration strategy
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.state_discretization = state_discretization
        self.rng = np.random.RandomState(seed)
        
        # Initialize exploration strategy
        if isinstance(exploration_strategy, str):
            strategy_kwargs["seed"] = seed
            self.exploration_strategy = create_exploration_strategy(
                exploration_strategy, **strategy_kwargs
            )
        else:
            self.exploration_strategy = exploration_strategy
        
        # Q-table for storing state-action values
        self.q_table: Dict[Tuple, np.ndarray] = {}
        
        # Statistics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.q_value_history: List[float] = []

    def discretize_state(self, state: Any) -> Tuple:
        """Discretize continuous state for tabular Q-learning.
        
        Args:
            state: Continuous state from environment
            
        Returns:
            Discretized state tuple
        """
        if self.state_discretization is None:
            # For discrete states, return as-is
            if isinstance(state, np.ndarray):
                return tuple(state.astype(int))
            return tuple(state) if isinstance(state, (list, tuple)) else (state,)
        
        # Discretize continuous state
        if isinstance(state, np.ndarray):
            # Simple binning approach - can be improved with more sophisticated methods
            discretized = []
            for i, value in enumerate(state):
                # Normalize to [0, 1] and then discretize
                normalized = (value + 1) / 2  # Assuming state is in [-1, 1]
                bin_idx = min(int(normalized * self.state_discretization), 
                            self.state_discretization - 1)
                discretized.append(bin_idx)
            return tuple(discretized)
        
        return tuple(state)

    def get_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for all actions in the given state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for all actions
        """
        discretized_state = self.discretize_state(state)
        
        if discretized_state not in self.q_table:
            self.q_table[discretized_state] = np.zeros(self.action_space_size)
        
        return self.q_table[discretized_state]

    def select_action(self, state: Any) -> int:
        """Select action using exploration strategy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        q_values = self.get_q_values(state)
        return self.exploration_strategy.select_action(
            q_values, state, self.action_space_size
        )

    def update(
        self, 
        state: Any, 
        action: int, 
        reward: float, 
        next_state: Any, 
        done: bool
    ) -> None:
        """Update Q-values using Q-learning rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Get current Q-value
        current_q_values = self.get_q_values(state)
        current_q_value = current_q_values[action]
        
        # Calculate target Q-value
        if done:
            target_q_value = reward
        else:
            next_q_values = self.get_q_values(next_state)
            target_q_value = reward + self.discount_factor * np.max(next_q_values)
        
        # Update Q-value using Q-learning rule
        new_q_value = current_q_value + self.learning_rate * (
            target_q_value - current_q_value
        )
        
        # Store updated Q-value
        discretized_state = self.discretize_state(state)
        self.q_table[discretized_state][action] = new_q_value
        
        # Update exploration strategy
        self.exploration_strategy.update(state, action, reward, next_state)
        
        # Track Q-value changes
        self.q_value_history.append(abs(new_q_value - current_q_value))

    def decay_exploration(self) -> None:
        """Decay exploration parameters."""
        self.exploration_strategy.decay()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary containing various statistics
        """
        stats = {
            "num_states": len(self.q_table),
            "total_q_updates": len(self.q_value_history),
            "avg_q_change": np.mean(self.q_value_history) if self.q_value_history else 0,
            "exploration_rate": getattr(self.exploration_strategy, "epsilon", None),
        }
        
        if self.episode_rewards:
            stats.update({
                "avg_reward": np.mean(self.episode_rewards),
                "std_reward": np.std(self.episode_rewards),
                "max_reward": np.max(self.episode_rewards),
                "min_reward": np.min(self.episode_rewards),
            })
        
        if self.episode_lengths:
            stats.update({
                "avg_episode_length": np.mean(self.episode_lengths),
                "std_episode_length": np.std(self.episode_lengths),
            })
        
        return stats

    def reset_statistics(self) -> None:
        """Reset episode statistics."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []

    def save_q_table(self, filepath: str) -> None:
        """Save Q-table to file.
        
        Args:
            filepath: Path to save Q-table
        """
        np.savez(filepath, **{str(k): v for k, v in self.q_table.items()})

    def load_q_table(self, filepath: str) -> None:
        """Load Q-table from file.
        
        Args:
            filepath: Path to load Q-table from
        """
        data = np.load(filepath)
        self.q_table = {eval(k): v for k, v in data.items()}


class DQNAgent:
    """Deep Q-Network agent with exploration strategies.
    
    This agent uses neural networks to approximate Q-values instead of
    tabular storage, enabling it to handle high-dimensional state spaces.
    """

    def __init__(
        self,
        state_dim: int,
        action_space_size: int,
        exploration_strategy: Union[str, ExplorationStrategy] = "epsilon_greedy",
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        device: Optional[str] = None,
        seed: Optional[int] = None,
        **strategy_kwargs
    ):
        """Initialize DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_space_size: Number of possible actions
            exploration_strategy: Strategy name or instance
            learning_rate: Learning rate for neural network
            discount_factor: Discount factor for future rewards
            device: Device to run neural network on
            seed: Random seed for reproducibility
            **strategy_kwargs: Additional arguments for exploration strategy
        """
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize exploration strategy
        if isinstance(exploration_strategy, str):
            strategy_kwargs["seed"] = seed
            self.exploration_strategy = create_exploration_strategy(
                exploration_strategy, **strategy_kwargs
            )
        else:
            self.exploration_strategy = exploration_strategy
        
        # Initialize neural network
        self.q_network = self._build_network()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Statistics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.loss_history: List[float] = []

    def _build_network(self) -> torch.nn.Module:
        """Build Q-network architecture."""
        return torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, self.action_space_size)
        ).to(self.device)

    def get_q_values(self, state: Any) -> np.ndarray:
        """Get Q-values for all actions in the given state.
        
        Args:
            state: Current state
            
        Returns:
            Q-values for all actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.cpu().numpy().flatten()

    def select_action(self, state: Any) -> int:
        """Select action using exploration strategy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        q_values = self.get_q_values(state)
        return self.exploration_strategy.select_action(
            q_values, state, self.action_space_size
        )

    def update(
        self, 
        state: Any, 
        action: int, 
        reward: float, 
        next_state: Any, 
        done: bool
    ) -> None:
        """Update Q-network using experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        # Get current Q-values
        current_q_values = self.q_network(state_tensor)
        current_q_value = current_q_values[0, action]
        
        # Calculate target Q-value
        with torch.no_grad():
            next_q_values = self.q_network(next_state_tensor)
            if done:
                target_q_value = torch.tensor(reward, device=self.device)
            else:
                target_q_value = torch.tensor(reward, device=self.device) + \
                               self.discount_factor * torch.max(next_q_values)
        
        # Calculate loss and update network
        loss = torch.nn.functional.mse_loss(current_q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Track loss
        self.loss_history.append(loss.item())
        
        # Update exploration strategy
        self.exploration_strategy.update(state, action, reward, next_state)

    def decay_exploration(self) -> None:
        """Decay exploration parameters."""
        self.exploration_strategy.decay()

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "total_updates": len(self.loss_history),
            "avg_loss": np.mean(self.loss_history) if self.loss_history else 0,
            "exploration_rate": getattr(self.exploration_strategy, "epsilon", None),
        }
        
        if self.episode_rewards:
            stats.update({
                "avg_reward": np.mean(self.episode_rewards),
                "std_reward": np.std(self.episode_rewards),
                "max_reward": np.max(self.episode_rewards),
                "min_reward": np.min(self.episode_rewards),
            })
        
        return stats

    def save_model(self, filepath: str) -> None:
        """Save Q-network model."""
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath: str) -> None:
        """Load Q-network model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
