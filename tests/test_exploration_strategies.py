"""Tests for exploration strategies."""

import pytest
import numpy as np
import torch

from src.algorithms.exploration_strategies import (
    EpsilonGreedyStrategy,
    UCBActionSelection,
    ThompsonSamplingStrategy,
    CuriosityDrivenStrategy,
    create_exploration_strategy,
)


class TestEpsilonGreedyStrategy:
    """Test epsilon-greedy exploration strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = EpsilonGreedyStrategy(epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.01)
        assert strategy.epsilon == 0.5
        assert strategy.epsilon_decay == 0.9
        assert strategy.epsilon_min == 0.01

    def test_action_selection_exploration(self):
        """Test action selection during exploration."""
        strategy = EpsilonGreedyStrategy(epsilon=1.0, seed=42)
        q_values = np.array([0.1, 0.9, 0.5])
        
        # With epsilon=1.0, should always explore (random action)
        actions = [strategy.select_action(q_values, None, 3) for _ in range(10)]
        assert all(0 <= action < 3 for action in actions)

    def test_action_selection_exploitation(self):
        """Test action selection during exploitation."""
        strategy = EpsilonGreedyStrategy(epsilon=0.0, seed=42)
        q_values = np.array([0.1, 0.9, 0.5])
        
        # With epsilon=0.0, should always exploit (best action)
        action = strategy.select_action(q_values, None, 3)
        assert action == 1  # Index of max value

    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        strategy = EpsilonGreedyStrategy(epsilon=1.0, epsilon_decay=0.5, epsilon_min=0.1)
        
        initial_epsilon = strategy.epsilon
        strategy.decay()
        
        assert strategy.epsilon == initial_epsilon * 0.5
        assert strategy.epsilon >= strategy.epsilon_min

    def test_epsilon_minimum(self):
        """Test that epsilon doesn't go below minimum."""
        strategy = EpsilonGreedyStrategy(epsilon=0.2, epsilon_decay=0.1, epsilon_min=0.1)
        
        strategy.decay()
        assert strategy.epsilon == strategy.epsilon_min


class TestUCBActionSelection:
    """Test UCB action selection strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = UCBActionSelection(c=2.0, seed=42)
        assert strategy.c == 2.0
        assert len(strategy.action_counts) == 0
        assert strategy.total_steps == 0

    def test_action_selection_first_step(self):
        """Test action selection on first step."""
        strategy = UCBActionSelection(c=2.0, seed=42)
        q_values = np.array([0.1, 0.9, 0.5])
        
        action = strategy.select_action(q_values, None, 3)
        assert 0 <= action < 3
        assert strategy.total_steps == 1

    def test_action_selection_with_counts(self):
        """Test action selection with existing counts."""
        strategy = UCBActionSelection(c=2.0, seed=42)
        q_values = np.array([0.1, 0.9, 0.5])
        
        # First action
        action1 = strategy.select_action(q_values, None, 3)
        strategy.update(None, action1, 1.0, None)
        
        # Second action
        action2 = strategy.select_action(q_values, None, 3)
        strategy.update(None, action2, 1.0, None)
        
        assert strategy.total_steps == 2
        assert len(strategy.action_counts) == 1  # One state

    def test_update_functionality(self):
        """Test update functionality."""
        strategy = UCBActionSelection(c=2.0, seed=42)
        
        strategy.update("state1", 0, 1.0, "next_state")
        strategy.update("state1", 1, 0.5, "next_state")
        
        state_key = strategy._state_to_key("state1")
        assert state_key in strategy.action_counts
        assert strategy.action_counts[state_key][0] == 1
        assert strategy.action_counts[state_key][1] == 1


class TestThompsonSamplingStrategy:
    """Test Thompson sampling strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = ThompsonSamplingStrategy(alpha=2.0, beta=3.0, seed=42)
        assert strategy.alpha == 2.0
        assert strategy.beta == 3.0
        assert len(strategy.action_stats) == 0

    def test_action_selection(self):
        """Test action selection."""
        strategy = ThompsonSamplingStrategy(alpha=1.0, beta=1.0, seed=42)
        q_values = np.array([0.1, 0.9, 0.5])
        
        action = strategy.select_action(q_values, None, 3)
        assert 0 <= action < 3

    def test_update_functionality(self):
        """Test update functionality."""
        strategy = ThompsonSamplingStrategy(alpha=1.0, beta=1.0, seed=42)
        
        # Update with positive reward
        strategy.update("state1", 0, 1.0, "next_state")
        
        state_key = strategy._state_to_key("state1")
        assert state_key in strategy.action_stats
        assert 0 in strategy.action_stats[state_key]
        
        alpha, beta = strategy.action_stats[state_key][0]
        assert alpha > strategy.alpha  # Should increase for positive reward
        assert beta == strategy.beta

    def test_state_key_conversion(self):
        """Test state to key conversion."""
        strategy = ThompsonSamplingStrategy(alpha=1.0, beta=1.0, seed=42)
        
        # Test numpy array
        state_array = np.array([1, 2, 3])
        key1 = strategy._state_to_key(state_array)
        assert isinstance(key1, tuple)
        assert key1 == (1, 2, 3)
        
        # Test list
        state_list = [1, 2, 3]
        key2 = strategy._state_to_key(state_list)
        assert key2 == (1, 2, 3)
        
        # Test scalar
        state_scalar = 42
        key3 = strategy._state_to_key(state_scalar)
        assert key3 == (42,)


class TestCuriosityDrivenStrategy:
    """Test curiosity-driven exploration strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = CuriosityDrivenStrategy(intrinsic_weight=0.2, learning_rate=0.01, seed=42)
        assert strategy.intrinsic_weight == 0.2
        assert strategy.learning_rate == 0.01
        assert isinstance(strategy.prediction_network, torch.nn.Module)

    def test_action_selection(self):
        """Test action selection."""
        strategy = CuriosityDrivenStrategy(intrinsic_weight=0.1, seed=42)
        q_values = np.array([0.1, 0.9, 0.5])
        
        action = strategy.select_action(q_values, None, 3)
        assert 0 <= action < 3

    def test_intrinsic_reward_calculation(self):
        """Test intrinsic reward calculation."""
        strategy = CuriosityDrivenStrategy(intrinsic_weight=0.1, seed=42)
        
        # First state should have maximum curiosity
        intrinsic_reward = strategy._get_intrinsic_reward(np.array([1, 2, 3, 4]))
        assert intrinsic_reward == 1.0

    def test_update_functionality(self):
        """Test update functionality."""
        strategy = CuriosityDrivenStrategy(intrinsic_weight=0.1, seed=42)
        
        state1 = np.array([1, 2, 3, 4])
        state2 = np.array([2, 3, 4, 5])
        
        strategy.update(state1, 0, 1.0, state2)
        assert len(strategy.state_buffer) == 1
        assert strategy.state_buffer[0] is state1


class TestExplorationStrategyFactory:
    """Test exploration strategy factory function."""

    def test_create_epsilon_greedy(self):
        """Test creating epsilon-greedy strategy."""
        strategy = create_exploration_strategy("epsilon_greedy", epsilon=0.5)
        assert isinstance(strategy, EpsilonGreedyStrategy)
        assert strategy.epsilon == 0.5

    def test_create_ucb(self):
        """Test creating UCB strategy."""
        strategy = create_exploration_strategy("ucb", c=3.0)
        assert isinstance(strategy, UCBActionSelection)
        assert strategy.c == 3.0

    def test_create_thompson(self):
        """Test creating Thompson sampling strategy."""
        strategy = create_exploration_strategy("thompson", alpha=2.0, beta=2.0)
        assert isinstance(strategy, ThompsonSamplingStrategy)
        assert strategy.alpha == 2.0
        assert strategy.beta == 2.0

    def test_create_curiosity(self):
        """Test creating curiosity-driven strategy."""
        strategy = create_exploration_strategy("curiosity", intrinsic_weight=0.2)
        assert isinstance(strategy, CuriosityDrivenStrategy)
        assert strategy.intrinsic_weight == 0.2

    def test_invalid_strategy(self):
        """Test creating invalid strategy raises error."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            create_exploration_strategy("invalid_strategy")


class TestExplorationStrategyIntegration:
    """Integration tests for exploration strategies."""

    def test_strategy_consistency(self):
        """Test that strategies maintain consistency across calls."""
        strategies = [
            EpsilonGreedyStrategy(epsilon=0.5, seed=42),
            UCBActionSelection(c=2.0, seed=42),
            ThompsonSamplingStrategy(alpha=1.0, beta=1.0, seed=42),
        ]
        
        q_values = np.array([0.1, 0.9, 0.5])
        state = np.array([1, 2, 3, 4])
        
        for strategy in strategies:
            # Multiple calls should be consistent
            actions = [strategy.select_action(q_values, state, 3) for _ in range(5)]
            
            # All actions should be valid
            assert all(0 <= action < 3 for action in actions)
            
            # Update should not break consistency
            strategy.update(state, actions[0], 1.0, state)
            new_action = strategy.select_action(q_values, state, 3)
            assert 0 <= new_action < 3

    def test_strategy_with_different_state_types(self):
        """Test strategies work with different state types."""
        strategies = [
            EpsilonGreedyStrategy(epsilon=0.5, seed=42),
            UCBActionSelection(c=2.0, seed=42),
            ThompsonSamplingStrategy(alpha=1.0, beta=1.0, seed=42),
        ]
        
        q_values = np.array([0.1, 0.9, 0.5])
        
        # Test different state types
        state_types = [
            np.array([1, 2, 3, 4]),
            [1, 2, 3, 4],
            (1, 2, 3, 4),
            42,
        ]
        
        for strategy in strategies:
            for state in state_types:
                action = strategy.select_action(q_values, state, 3)
                assert 0 <= action < 3
                
                strategy.update(state, action, 1.0, state)
                new_action = strategy.select_action(q_values, state, 3)
                assert 0 <= new_action < 3
