"""Tests for Q-learning agents."""

import pytest
import numpy as np
import torch

from src.algorithms.q_learning import QLearningAgent, DQNAgent
from src.algorithms.exploration_strategies import EpsilonGreedyStrategy


class TestQLearningAgent:
    """Test Q-learning agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = QLearningAgent(action_space_size=4, seed=42)
        assert agent.action_space_size == 4
        assert len(agent.q_table) == 0
        assert isinstance(agent.exploration_strategy, EpsilonGreedyStrategy)

    def test_discretize_state_discrete(self):
        """Test state discretization for discrete states."""
        agent = QLearningAgent(action_space_size=2, state_discretization=None)
        
        # Test numpy array
        state = np.array([1, 2, 3])
        discretized = agent.discretize_state(state)
        assert discretized == (1, 2, 3)
        
        # Test list
        state = [1, 2, 3]
        discretized = agent.discretize_state(state)
        assert discretized == (1, 2, 3)
        
        # Test scalar
        state = 42
        discretized = agent.discretize_state(state)
        assert discretized == (42,)

    def test_discretize_state_continuous(self):
        """Test state discretization for continuous states."""
        agent = QLearningAgent(action_space_size=2, state_discretization=10)
        
        # Test continuous state
        state = np.array([0.5, -0.3, 0.8, -0.1])
        discretized = agent.discretize_state(state)
        
        # Should be discretized to bins
        assert len(discretized) == 4
        assert all(0 <= bin_idx < 10 for bin_idx in discretized)

    def test_get_q_values_new_state(self):
        """Test getting Q-values for new state."""
        agent = QLearningAgent(action_space_size=3, seed=42)
        state = np.array([1, 2, 3])
        
        q_values = agent.get_q_values(state)
        assert len(q_values) == 3
        assert np.allclose(q_values, 0.0)
        assert len(agent.q_table) == 1

    def test_get_q_values_existing_state(self):
        """Test getting Q-values for existing state."""
        agent = QLearningAgent(action_space_size=3, seed=42)
        state = np.array([1, 2, 3])
        
        # First call
        q_values1 = agent.get_q_values(state)
        
        # Modify Q-values
        discretized_state = agent.discretize_state(state)
        agent.q_table[discretized_state][0] = 1.0
        
        # Second call
        q_values2 = agent.get_q_values(state)
        assert q_values2[0] == 1.0
        assert q_values2[1] == 0.0
        assert q_values2[2] == 0.0

    def test_select_action(self):
        """Test action selection."""
        agent = QLearningAgent(action_space_size=3, seed=42)
        state = np.array([1, 2, 3])
        
        action = agent.select_action(state)
        assert 0 <= action < 3

    def test_update_q_values(self):
        """Test Q-value updates."""
        agent = QLearningAgent(
            action_space_size=3,
            learning_rate=0.1,
            discount_factor=0.9,
            seed=42
        )
        
        state = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_state = np.array([2, 3, 4])
        
        # Initial Q-value
        initial_q = agent.get_q_values(state)[action]
        
        # Update Q-value
        agent.update(state, action, reward, next_state, False)
        
        # Check Q-value changed
        new_q = agent.get_q_values(state)[action]
        assert new_q != initial_q

    def test_update_q_values_terminal(self):
        """Test Q-value updates for terminal states."""
        agent = QLearningAgent(
            action_space_size=3,
            learning_rate=0.1,
            discount_factor=0.9,
            seed=42
        )
        
        state = np.array([1, 2, 3])
        action = 1
        reward = 1.0
        next_state = np.array([2, 3, 4])
        
        # Update for terminal state
        agent.update(state, action, reward, next_state, True)
        
        # Q-value should be updated
        q_value = agent.get_q_values(state)[action]
        assert q_value == reward  # Terminal state update

    def test_decay_exploration(self):
        """Test exploration decay."""
        agent = QLearningAgent(action_space_size=3, seed=42)
        initial_epsilon = agent.exploration_strategy.epsilon
        
        agent.decay_exploration()
        
        assert agent.exploration_strategy.epsilon < initial_epsilon

    def test_get_statistics(self):
        """Test statistics collection."""
        agent = QLearningAgent(action_space_size=3, seed=42)
        
        # Add some experience
        state = np.array([1, 2, 3])
        agent.update(state, 1, 1.0, state, False)
        
        stats = agent.get_statistics()
        assert "num_states" in stats
        assert "total_q_updates" in stats
        assert "avg_q_change" in stats
        assert stats["num_states"] == 1

    def test_reset_statistics(self):
        """Test statistics reset."""
        agent = QLearningAgent(action_space_size=3, seed=42)
        
        # Add some statistics
        agent.episode_rewards = [1.0, 2.0, 3.0]
        agent.episode_lengths = [10, 20, 30]
        agent.q_value_history = [0.1, 0.2, 0.3]
        
        agent.reset_statistics()
        
        assert len(agent.episode_rewards) == 0
        assert len(agent.episode_lengths) == 0
        assert len(agent.q_value_history) == 0


class TestDQNAgent:
    """Test Deep Q-Network agent."""

    def test_initialization(self):
        """Test agent initialization."""
        agent = DQNAgent(state_dim=4, action_space_size=2, seed=42)
        assert agent.state_dim == 4
        assert agent.action_space_size == 2
        assert isinstance(agent.q_network, torch.nn.Module)

    def test_get_q_values(self):
        """Test getting Q-values."""
        agent = DQNAgent(state_dim=4, action_space_size=3, seed=42)
        state = np.array([1, 2, 3, 4])
        
        q_values = agent.get_q_values(state)
        assert len(q_values) == 3
        assert isinstance(q_values, np.ndarray)

    def test_select_action(self):
        """Test action selection."""
        agent = DQNAgent(state_dim=4, action_space_size=3, seed=42)
        state = np.array([1, 2, 3, 4])
        
        action = agent.select_action(state)
        assert 0 <= action < 3

    def test_update(self):
        """Test network update."""
        agent = DQNAgent(
            state_dim=4,
            action_space_size=3,
            learning_rate=0.01,
            seed=42
        )
        
        state = np.array([1, 2, 3, 4])
        action = 1
        reward = 1.0
        next_state = np.array([2, 3, 4, 5])
        
        # Get initial loss
        initial_loss = agent.loss_history[-1] if agent.loss_history else 0
        
        # Update network
        agent.update(state, action, reward, next_state, False)
        
        # Check loss was recorded
        assert len(agent.loss_history) > 0

    def test_update_terminal(self):
        """Test network update for terminal states."""
        agent = DQNAgent(
            state_dim=4,
            action_space_size=3,
            learning_rate=0.01,
            seed=42
        )
        
        state = np.array([1, 2, 3, 4])
        action = 1
        reward = 1.0
        next_state = np.array([2, 3, 4, 5])
        
        # Update for terminal state
        agent.update(state, action, reward, next_state, True)
        
        # Should not crash
        assert len(agent.loss_history) > 0

    def test_decay_exploration(self):
        """Test exploration decay."""
        agent = DQNAgent(state_dim=4, action_space_size=3, seed=42)
        initial_epsilon = agent.exploration_strategy.epsilon
        
        agent.decay_exploration()
        
        assert agent.exploration_strategy.epsilon < initial_epsilon

    def test_get_statistics(self):
        """Test statistics collection."""
        agent = DQNAgent(state_dim=4, action_space_size=3, seed=42)
        
        # Add some experience
        state = np.array([1, 2, 3, 4])
        agent.update(state, 1, 1.0, state, False)
        
        stats = agent.get_statistics()
        assert "total_updates" in stats
        assert "avg_loss" in stats
        assert "exploration_rate" in stats

    def test_save_load_model(self):
        """Test model saving and loading."""
        agent = DQNAgent(state_dim=4, action_space_size=3, seed=42)
        
        # Save model
        agent.save_model("test_model.pth")
        
        # Create new agent and load model
        new_agent = DQNAgent(state_dim=4, action_space_size=3, seed=42)
        new_agent.load_model("test_model.pth")
        
        # Models should be equivalent
        state = np.array([1, 2, 3, 4])
        q_values1 = agent.get_q_values(state)
        q_values2 = new_agent.get_q_values(state)
        
        assert np.allclose(q_values1, q_values2)
        
        # Clean up
        import os
        os.remove("test_model.pth")


class TestAgentIntegration:
    """Integration tests for agents."""

    def test_agent_consistency(self):
        """Test that agents maintain consistency across calls."""
        agents = [
            QLearningAgent(action_space_size=3, seed=42),
            DQNAgent(state_dim=4, action_space_size=3, seed=42),
        ]
        
        state = np.array([1, 2, 3, 4])
        
        for agent in agents:
            # Multiple calls should be consistent
            actions = [agent.select_action(state) for _ in range(5)]
            
            # All actions should be valid
            assert all(0 <= action < 3 for action in actions)
            
            # Update should not break consistency
            agent.update(state, actions[0], 1.0, state, False)
            new_action = agent.select_action(state)
            assert 0 <= new_action < 3

    def test_agent_with_different_exploration_strategies(self):
        """Test agents work with different exploration strategies."""
        from src.algorithms.exploration_strategies import create_exploration_strategy
        
        strategies = ["epsilon_greedy", "ucb", "thompson"]
        
        for strategy_name in strategies:
            agent = QLearningAgent(
                action_space_size=3,
                exploration_strategy=strategy_name,
                seed=42
            )
            
            state = np.array([1, 2, 3])
            action = agent.select_action(state)
            assert 0 <= action < 3
            
            agent.update(state, action, 1.0, state, False)
            new_action = agent.select_action(state)
            assert 0 <= new_action < 3
