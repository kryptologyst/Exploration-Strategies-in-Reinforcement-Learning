"""Core exploration strategies for reinforcement learning.

This module implements various exploration strategies including epsilon-greedy,
UCB, Thompson sampling, and curiosity-driven exploration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import Env


class ExplorationStrategy(ABC):
    """Abstract base class for exploration strategies."""

    @abstractmethod
    def select_action(
        self, 
        q_values: np.ndarray, 
        state: Any, 
        action_space_size: int
    ) -> int:
        """Select an action using the exploration strategy.
        
        Args:
            q_values: Current Q-values for all actions
            state: Current state (for state-dependent strategies)
            action_space_size: Size of the action space
            
        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """Update the exploration strategy based on experience.
        
        Args:
            state: Previous state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        pass

    @abstractmethod
    def decay(self) -> None:
        """Decay exploration parameters (e.g., epsilon)."""
        pass


class EpsilonGreedyStrategy(ExplorationStrategy):
    """Epsilon-greedy exploration strategy.
    
    With probability epsilon, selects a random action. Otherwise, selects
    the action with the highest Q-value.
    """

    def __init__(
        self, 
        epsilon: float = 1.0, 
        epsilon_decay: float = 0.995, 
        epsilon_min: float = 0.01,
        seed: Optional[int] = None
    ):
        """Initialize epsilon-greedy strategy.
        
        Args:
            epsilon: Initial exploration rate
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
            seed: Random seed for reproducibility
        """
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.rng = np.random.RandomState(seed)

    def select_action(
        self, 
        q_values: np.ndarray, 
        state: Any, 
        action_space_size: int
    ) -> int:
        """Select action using epsilon-greedy strategy."""
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, action_space_size)
        else:
            return int(np.argmax(q_values))

    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """No update needed for epsilon-greedy."""
        pass

    def decay(self) -> None:
        """Decay epsilon value."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class UCBActionSelection(ExplorationStrategy):
    """Upper Confidence Bound (UCB) action selection strategy.
    
    Balances exploitation and exploration by considering both the estimated
    Q-value and the uncertainty (confidence bound) of each action.
    """

    def __init__(
        self, 
        c: float = 2.0, 
        seed: Optional[int] = None
    ):
        """Initialize UCB strategy.
        
        Args:
            c: Exploration parameter (higher = more exploration)
            seed: Random seed for reproducibility
        """
        self.c = c
        self.rng = np.random.RandomState(seed)
        self.action_counts: Dict[Tuple, np.ndarray] = {}
        self.total_steps = 0

    def select_action(
        self, 
        q_values: np.ndarray, 
        state: Any, 
        action_space_size: int
    ) -> int:
        """Select action using UCB strategy."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.action_counts:
            self.action_counts[state_key] = np.zeros(action_space_size)
        
        counts = self.action_counts[state_key]
        self.total_steps += 1
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        
        # Calculate UCB values
        confidence_bounds = self.c * np.sqrt(np.log(self.total_steps) / counts)
        ucb_values = q_values + confidence_bounds
        
        return int(np.argmax(ucb_values))

    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """Update action counts."""
        state_key = self._state_to_key(state)
        if state_key not in self.action_counts:
            self.action_counts[state_key] = np.zeros(len(self.action_counts.get(state_key, [])))
        self.action_counts[state_key][action] += 1

    def decay(self) -> None:
        """No decay for UCB."""
        pass

    def _state_to_key(self, state: Any) -> Tuple:
        """Convert state to hashable key."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, (list, tuple)):
            return tuple(state)
        else:
            return (state,)


class ThompsonSamplingStrategy(ExplorationStrategy):
    """Thompson sampling exploration strategy.
    
    Uses Bayesian inference to sample actions from posterior distributions
    of Q-values, naturally balancing exploration and exploitation.
    """

    def __init__(
        self, 
        alpha: float = 1.0, 
        beta: float = 1.0,
        seed: Optional[int] = None
    ):
        """Initialize Thompson sampling strategy.
        
        Args:
            alpha: Prior alpha parameter for Beta distribution
            beta: Prior beta parameter for Beta distribution
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.beta = beta
        self.rng = np.random.RandomState(seed)
        self.action_stats: Dict[Tuple, Dict[int, Tuple[float, float]]] = {}

    def select_action(
        self, 
        q_values: np.ndarray, 
        state: Any, 
        action_space_size: int
    ) -> int:
        """Select action using Thompson sampling."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.action_stats:
            self.action_stats[state_key] = {}
        
        # Sample from Beta distributions for each action
        sampled_values = np.zeros(action_space_size)
        
        for action in range(action_space_size):
            if action in self.action_stats[state_key]:
                alpha_a, beta_a = self.action_stats[state_key][action]
            else:
                alpha_a, beta_a = self.alpha, self.beta
            
            # Sample from Beta distribution
            sampled_values[action] = self.rng.beta(alpha_a, beta_a)
        
        return int(np.argmax(sampled_values))

    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """Update Beta distribution parameters."""
        state_key = self._state_to_key(state)
        
        if state_key not in self.action_stats:
            self.action_stats[state_key] = {}
        
        if action not in self.action_stats[state_key]:
            self.action_stats[state_key][action] = (self.alpha, self.beta)
        
        alpha_a, beta_a = self.action_stats[state_key][action]
        
        # Update Beta parameters based on reward (normalized to [0,1])
        normalized_reward = max(0, min(1, (reward + 1) / 2))  # Assuming rewards in [-1, 1]
        
        if normalized_reward > 0.5:  # Positive reward
            alpha_a += 1
        else:  # Negative or neutral reward
            beta_a += 1
        
        self.action_stats[state_key][action] = (alpha_a, beta_a)

    def decay(self) -> None:
        """No decay for Thompson sampling."""
        pass

    def _state_to_key(self, state: Any) -> Tuple:
        """Convert state to hashable key."""
        if isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, (list, tuple)):
            return tuple(state)
        else:
            return (state,)


class CuriosityDrivenStrategy(ExplorationStrategy):
    """Curiosity-driven exploration using intrinsic motivation.
    
    Uses prediction error as intrinsic reward to encourage exploration
    of novel states and actions.
    """

    def __init__(
        self, 
        intrinsic_weight: float = 0.1,
        learning_rate: float = 0.001,
        seed: Optional[int] = None
    ):
        """Initialize curiosity-driven strategy.
        
        Args:
            intrinsic_weight: Weight for intrinsic reward
            learning_rate: Learning rate for prediction network
            seed: Random seed for reproducibility
        """
        self.intrinsic_weight = intrinsic_weight
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed)
        
        # Simple neural network for state prediction
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_network = self._build_prediction_network()
        self.optimizer = torch.optim.Adam(
            self.prediction_network.parameters(), 
            lr=learning_rate
        )
        
        self.state_buffer = []
        self.prediction_errors = []

    def _build_prediction_network(self) -> nn.Module:
        """Build a simple prediction network."""
        return nn.Sequential(
            nn.Linear(4, 64),  # CartPole state size
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # Predict next state
        ).to(self.device)

    def select_action(
        self, 
        q_values: np.ndarray, 
        state: Any, 
        action_space_size: int
    ) -> int:
        """Select action using curiosity-driven strategy."""
        # Add intrinsic reward to Q-values
        intrinsic_reward = self._get_intrinsic_reward(state)
        modified_q_values = q_values + self.intrinsic_weight * intrinsic_reward
        
        return int(np.argmax(modified_q_values))

    def update(self, state: Any, action: int, reward: float, next_state: Any) -> None:
        """Update prediction network and intrinsic rewards."""
        if len(self.state_buffer) > 0:
            # Train prediction network
            prev_state = self.state_buffer[-1]
            self._train_prediction_network(prev_state, next_state)
        
        self.state_buffer.append(state)

    def decay(self) -> None:
        """No decay for curiosity-driven strategy."""
        pass

    def _get_intrinsic_reward(self, state: Any) -> float:
        """Calculate intrinsic reward based on prediction error."""
        if len(self.state_buffer) == 0:
            return 1.0  # Maximum curiosity for first state
        
        prev_state = self.state_buffer[-1]
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predicted_next_state = self.prediction_network(state_tensor)
            prediction_error = torch.norm(predicted_next_state - state_tensor).item()
        
        return prediction_error

    def _train_prediction_network(self, prev_state: Any, next_state: Any) -> None:
        """Train the prediction network."""
        prev_tensor = torch.FloatTensor(prev_state).unsqueeze(0).to(self.device)
        next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        
        predicted_next = self.prediction_network(prev_tensor)
        loss = F.mse_loss(predicted_next, next_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def create_exploration_strategy(
    strategy_name: str, 
    **kwargs
) -> ExplorationStrategy:
    """Factory function to create exploration strategies.
    
    Args:
        strategy_name: Name of the strategy ('epsilon_greedy', 'ucb', 'thompson', 'curiosity')
        **kwargs: Additional arguments for the strategy
        
    Returns:
        ExplorationStrategy instance
        
    Raises:
        ValueError: If strategy_name is not recognized
    """
    strategies = {
        'epsilon_greedy': EpsilonGreedyStrategy,
        'ucb': UCBActionSelection,
        'thompson': ThompsonSamplingStrategy,
        'curiosity': CuriosityDrivenStrategy,
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs)
