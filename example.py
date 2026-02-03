#!/usr/bin/env python3
"""Simple example demonstrating exploration strategies in RL."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
from algorithms.q_learning import QLearningAgent
from algorithms.exploration_strategies import create_exploration_strategy
from evaluation.trainer import Trainer, TrainingConfig
from utils.utils import set_seed


def main():
    """Run a simple example comparing exploration strategies."""
    print("Exploration Strategies in RL - Simple Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create training configuration
    config = TrainingConfig(
        env_name="CartPole-v1",
        num_episodes=200,  # Reduced for quick demo
        max_steps_per_episode=500,
        eval_frequency=50,
        num_eval_episodes=5,
        seed=42,
        output_dir="example_outputs"
    )
    
    print(f"Environment: {config.env_name}")
    print(f"Episodes: {config.num_episodes}")
    print(f"Evaluation frequency: {config.eval_frequency}")
    print()
    
    # Test different exploration strategies
    strategies = ["epsilon_greedy", "ucb", "thompson"]
    results = {}
    
    for strategy_name in strategies:
        print(f"Testing {strategy_name} strategy...")
        
        # Create agent
        agent = QLearningAgent(
            action_space_size=2,  # CartPole has 2 actions
            exploration_strategy=strategy_name,
            learning_rate=0.1,
            discount_factor=0.99,
            seed=42
        )
        
        # Train agent
        trainer = Trainer(agent, config, f"example_outputs/{strategy_name}")
        stats = trainer.train()
        
        # Store results
        final_eval_reward = stats["eval_rewards"][-1] if stats["eval_rewards"] else 0
        avg_training_reward = np.mean(stats["episode_rewards"])
        
        results[strategy_name] = {
            "final_eval_reward": final_eval_reward,
            "avg_training_reward": avg_training_reward,
            "max_training_reward": np.max(stats["episode_rewards"]),
            "avg_episode_length": np.mean(stats["episode_lengths"]),
        }
        
        print(f"  Final evaluation reward: {final_eval_reward:.2f}")
        print(f"  Average training reward: {avg_training_reward:.2f}")
        print()
    
    # Print summary
    print("Summary Results:")
    print("-" * 30)
    
    for strategy, result in results.items():
        print(f"{strategy.replace('_', ' ').title()}:")
        print(f"  Final eval reward: {result['final_eval_reward']:.2f}")
        print(f"  Avg training reward: {result['avg_training_reward']:.2f}")
        print(f"  Max training reward: {result['max_training_reward']:.2f}")
        print(f"  Avg episode length: {result['avg_episode_length']:.1f}")
        print()
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k]["final_eval_reward"])
    best_reward = results[best_strategy]["final_eval_reward"]
    
    print(f"Best performing strategy: {best_strategy.replace('_', ' ').title()}")
    print(f"Best final evaluation reward: {best_reward:.2f}")
    
    print("\nExample completed successfully!")
    print("Check the 'example_outputs' directory for detailed results and plots.")


if __name__ == "__main__":
    main()
