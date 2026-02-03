"""Main training script for exploration strategies comparison."""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from omegaconf import OmegaConf

from algorithms.q_learning import DQNAgent, QLearningAgent
from evaluation.trainer import Evaluator, Trainer, TrainingConfig
from utils.config import ExperimentConfig, create_default_configs
from utils.utils import set_seed


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agents with exploration strategies")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--strategy", type=str, default="epsilon_greedy", 
                       choices=["epsilon_greedy", "ucb", "thompson", "curiosity"],
                       help="Exploration strategy to use")
    parser.add_argument("--agent_type", type=str, default="tabular",
                       choices=["tabular", "dqn"],
                       help="Type of agent to use")
    parser.add_argument("--num_episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during training")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple strategies")
    parser.add_argument("--num_runs", type=int, default=5,
                       help="Number of runs for comparison")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare:
        # Compare multiple strategies
        compare_strategies(args)
    else:
        # Train single strategy
        train_single_strategy(args)


def train_single_strategy(args):
    """Train a single exploration strategy."""
    print(f"Training {args.strategy} strategy with {args.agent_type} agent...")
    
    # Create training configuration
    config = TrainingConfig(
        env_name="CartPole-v1",
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        render=args.render
    )
    
    # Create agent
    if args.agent_type == "tabular":
        agent = QLearningAgent(
            action_space_size=2,  # CartPole has 2 actions
            exploration_strategy=args.strategy,
            seed=args.seed
        )
    else:
        agent = DQNAgent(
            state_dim=4,  # CartPole state dimension
            action_space_size=2,
            exploration_strategy=args.strategy,
            seed=args.seed
        )
    
    # Train agent
    trainer = Trainer(agent, config, args.output_dir)
    stats = trainer.train()
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Print final statistics
    print("\nTraining completed!")
    print(f"Final evaluation reward: {stats['eval_rewards'][-1]:.2f}")
    print(f"Average training reward: {np.mean(stats['episode_rewards']):.2f}")
    print(f"Max training reward: {np.max(stats['episode_rewards']):.2f}")


def compare_strategies(args):
    """Compare multiple exploration strategies."""
    print("Comparing exploration strategies...")
    
    # Create evaluation configuration
    config = TrainingConfig(
        env_name="CartPole-v1",
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        render=args.render
    )
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Define strategies to compare
    strategies = ["epsilon_greedy", "ucb", "thompson", "curiosity"]
    
    # Run comparison
    results_df = evaluator.compare_strategies(
        strategies=strategies,
        agent_type=args.agent_type,
        num_runs=args.num_runs
    )
    
    # Plot comparison results
    evaluator.plot_comparison(results_df)
    
    # Generate and save report
    report = evaluator.generate_report(results_df)
    
    report_path = os.path.join(args.output_dir, "comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"\nComparison completed! Report saved to {report_path}")
    
    # Print summary
    print("\nSummary:")
    summary = results_df.groupby("strategy")["final_eval_reward"].agg(["mean", "std"])
    for strategy, row in summary.iterrows():
        print(f"{strategy}: {row['mean']:.2f} ± {row['std']:.2f}")


if __name__ == "__main__":
    main()
