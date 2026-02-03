"""Training and evaluation framework for exploration strategies."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from ..algorithms.q_learning import DQNAgent, QLearningAgent
from ..algorithms.exploration_strategies import create_exploration_strategy


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        num_episodes: int = 1000,
        max_steps_per_episode: int = 500,
        eval_frequency: int = 100,
        num_eval_episodes: int = 10,
        save_frequency: int = 500,
        seed: Optional[int] = None,
        render: bool = False,
        **kwargs
    ):
        """Initialize training configuration.
        
        Args:
            env_name: Name of the environment
            num_episodes: Number of training episodes
            max_steps_per_episode: Maximum steps per episode
            eval_frequency: Frequency of evaluation
            num_eval_episodes: Number of episodes for evaluation
            save_frequency: Frequency of saving checkpoints
            seed: Random seed
            render: Whether to render environment
            **kwargs: Additional configuration parameters
        """
        self.env_name = env_name
        self.num_episodes = num_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.eval_frequency = eval_frequency
        self.num_eval_episodes = num_eval_episodes
        self.save_frequency = save_frequency
        self.seed = seed
        self.render = render
        
        # Update with additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)


class Trainer:
    """Trainer class for RL agents with exploration strategies."""
    
    def __init__(
        self,
        agent: Union[QLearningAgent, DQNAgent],
        config: TrainingConfig,
        output_dir: str = "outputs"
    ):
        """Initialize trainer.
        
        Args:
            agent: RL agent to train
            config: Training configuration
            output_dir: Directory to save outputs
        """
        self.agent = agent
        self.config = config
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize environment
        self.env = gym.make(config.env_name)
        if config.seed is not None:
            self.env.reset(seed=config.seed)
        
        # Training statistics
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "eval_rewards": [],
            "eval_std": [],
            "exploration_rates": [],
            "timestamps": [],
        }
        
        # Set random seeds
        if config.seed is not None:
            np.random.seed(config.seed)
            self.env.reset(seed=config.seed)

    def train(self) -> Dict[str, List[float]]:
        """Train the agent.
        
        Returns:
            Dictionary containing training statistics
        """
        print(f"Starting training for {self.config.num_episodes} episodes...")
        print(f"Environment: {self.config.env_name}")
        print(f"Agent type: {type(self.agent).__name__}")
        print(f"Exploration strategy: {type(self.agent.exploration_strategy).__name__}")
        
        start_time = time.time()
        
        for episode in tqdm(range(self.config.num_episodes), desc="Training"):
            episode_reward, episode_length = self._train_episode(episode)
            
            # Record statistics
            self.training_stats["episode_rewards"].append(episode_reward)
            self.training_stats["episode_lengths"].append(episode_length)
            self.training_stats["timestamps"].append(time.time() - start_time)
            
            # Record exploration rate
            exploration_rate = getattr(
                self.agent.exploration_strategy, "epsilon", None
            )
            self.training_stats["exploration_rates"].append(exploration_rate)
            
            # Evaluation
            if episode % self.config.eval_frequency == 0:
                eval_reward, eval_std = self._evaluate()
                self.training_stats["eval_rewards"].append(eval_reward)
                self.training_stats["eval_std"].append(eval_std)
                
                print(f"Episode {episode}: "
                      f"Reward={episode_reward:.2f}, "
                      f"Length={episode_length}, "
                      f"Eval={eval_reward:.2f}±{eval_std:.2f}, "
                      f"Exploration={exploration_rate:.3f}")
            
            # Save checkpoint
            if episode % self.config.save_frequency == 0:
                self._save_checkpoint(episode)
        
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        return self.training_stats

    def _train_episode(self, episode: int) -> Tuple[float, int]:
        """Train for one episode.
        
        Args:
            episode: Episode number
            
        Returns:
            Tuple of (episode_reward, episode_length)
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.max_steps_per_episode):
            # Select action
            action = self.agent.select_action(state)
            
            # Take step
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Update agent
            self.agent.update(state, action, reward, next_state, done)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Render if requested
            if self.config.render:
                self.env.render()
            
            if done:
                break
        
        # Decay exploration
        self.agent.decay_exploration()
        
        return episode_reward, episode_length

    def _evaluate(self) -> Tuple[float, float]:
        """Evaluate the agent.
        
        Returns:
            Tuple of (mean_reward, std_reward)
        """
        eval_rewards = []
        
        for _ in range(self.config.num_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            
            for _ in range(self.config.max_steps_per_episode):
                # Use greedy action selection for evaluation
                q_values = self.agent.get_q_values(state)
                action = np.argmax(q_values)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards), np.std(eval_rewards)

    def _save_checkpoint(self, episode: int) -> None:
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.output_dir, f"checkpoint_episode_{episode}.pkl"
        )
        
        # Save agent-specific data
        if isinstance(self.agent, QLearningAgent):
            self.agent.save_q_table(checkpoint_path.replace('.pkl', '_qtable.npz'))
        elif isinstance(self.agent, DQNAgent):
            self.agent.save_model(checkpoint_path.replace('.pkl', '_model.pth'))
        
        # Save training statistics
        pd.DataFrame(self.training_stats).to_csv(
            os.path.join(self.output_dir, "training_stats.csv"), index=False
        )

    def plot_training_curves(self, save_path: Optional[str] = None) -> None:
        """Plot training curves.
        
        Args:
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_stats["episode_rewards"])
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)
        
        # Evaluation rewards
        if self.training_stats["eval_rewards"]:
            eval_episodes = np.arange(0, len(self.training_stats["eval_rewards"])) * self.config.eval_frequency
            axes[0, 1].errorbar(
                eval_episodes, 
                self.training_stats["eval_rewards"],
                yerr=self.training_stats["eval_std"],
                capsize=5
            )
            axes[0, 1].set_title("Evaluation Rewards")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Reward")
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.training_stats["episode_lengths"])
        axes[1, 0].set_title("Episode Lengths")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Length")
        axes[1, 0].grid(True)
        
        # Exploration rate
        if self.training_stats["exploration_rates"][0] is not None:
            axes[1, 1].plot(self.training_stats["exploration_rates"])
            axes[1, 1].set_title("Exploration Rate")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Epsilon")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "training_curves.png"), 
                       dpi=300, bbox_inches='tight')
        
        plt.show()


class Evaluator:
    """Evaluator class for comparing different exploration strategies."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize evaluator.
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.results = {}

    def compare_strategies(
        self,
        strategies: List[str],
        agent_type: str = "tabular",
        num_runs: int = 5,
        **strategy_kwargs
    ) -> pd.DataFrame:
        """Compare multiple exploration strategies.
        
        Args:
            strategies: List of strategy names to compare
            agent_type: Type of agent ('tabular' or 'dqn')
            num_runs: Number of independent runs
            **strategy_kwargs: Additional strategy parameters
            
        Returns:
            DataFrame with comparison results
        """
        print(f"Comparing {len(strategies)} strategies with {num_runs} runs each...")
        
        all_results = []
        
        for strategy in strategies:
            print(f"\nTesting strategy: {strategy}")
            
            for run in range(num_runs):
                print(f"Run {run + 1}/{num_runs}")
                
                # Create agent
                if agent_type == "tabular":
                    agent = QLearningAgent(
                        action_space_size=2,  # CartPole has 2 actions
                        exploration_strategy=strategy,
                        seed=self.config.seed + run,
                        **strategy_kwargs
                    )
                else:
                    agent = DQNAgent(
                        state_dim=4,  # CartPole state dimension
                        action_space_size=2,
                        exploration_strategy=strategy,
                        seed=self.config.seed + run,
                        **strategy_kwargs
                    )
                
                # Train agent
                trainer = Trainer(agent, self.config, f"outputs/{strategy}_run_{run}")
                stats = trainer.train()
                
                # Record results
                final_eval_reward = stats["eval_rewards"][-1] if stats["eval_rewards"] else 0
                final_eval_std = stats["eval_std"][-1] if stats["eval_std"] else 0
                
                all_results.append({
                    "strategy": strategy,
                    "run": run,
                    "final_eval_reward": final_eval_reward,
                    "final_eval_std": final_eval_std,
                    "avg_training_reward": np.mean(stats["episode_rewards"]),
                    "std_training_reward": np.std(stats["episode_rewards"]),
                    "max_training_reward": np.max(stats["episode_rewards"]),
                    "avg_episode_length": np.mean(stats["episode_lengths"]),
                })
        
        return pd.DataFrame(all_results)

    def plot_comparison(self, results_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot comparison results.
        
        Args:
            results_df: DataFrame with comparison results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Final evaluation rewards
        sns.boxplot(data=results_df, x="strategy", y="final_eval_reward", ax=axes[0, 0])
        axes[0, 0].set_title("Final Evaluation Rewards")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Average training rewards
        sns.boxplot(data=results_df, x="strategy", y="avg_training_reward", ax=axes[0, 1])
        axes[0, 1].set_title("Average Training Rewards")
        axes[0, 1].set_ylabel("Reward")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Episode lengths
        sns.boxplot(data=results_df, x="strategy", y="avg_episode_length", ax=axes[1, 0])
        axes[1, 0].set_title("Average Episode Lengths")
        axes[1, 0].set_ylabel("Length")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training reward variance
        sns.boxplot(data=results_df, x="strategy", y="std_training_reward", ax=axes[1, 1])
        axes[1, 1].set_title("Training Reward Variance")
        axes[1, 1].set_ylabel("Standard Deviation")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig("strategy_comparison.png", dpi=300, bbox_inches='tight')
        
        plt.show()

    def generate_report(self, results_df: pd.DataFrame) -> str:
        """Generate a comparison report.
        
        Args:
            results_df: DataFrame with comparison results
            
        Returns:
            Formatted report string
        """
        report = "# Exploration Strategy Comparison Report\n\n"
        
        # Summary statistics
        summary = results_df.groupby("strategy").agg({
            "final_eval_reward": ["mean", "std"],
            "avg_training_reward": ["mean", "std"],
            "avg_episode_length": ["mean", "std"],
        }).round(3)
        
        report += "## Summary Statistics\n\n"
        report += summary.to_string()
        report += "\n\n"
        
        # Best performing strategy
        best_strategy = results_df.groupby("strategy")["final_eval_reward"].mean().idxmax()
        best_reward = results_df.groupby("strategy")["final_eval_reward"].mean().max()
        
        report += f"## Best Performing Strategy\n\n"
        report += f"**{best_strategy}** achieved the highest average evaluation reward of {best_reward:.3f}.\n\n"
        
        # Detailed analysis
        report += "## Detailed Analysis\n\n"
        for strategy in results_df["strategy"].unique():
            strategy_data = results_df[results_df["strategy"] == strategy]
            report += f"### {strategy}\n"
            report += f"- Average final evaluation reward: {strategy_data['final_eval_reward'].mean():.3f} ± {strategy_data['final_eval_reward'].std():.3f}\n"
            report += f"- Average training reward: {strategy_data['avg_training_reward'].mean():.3f} ± {strategy_data['avg_training_reward'].std():.3f}\n"
            report += f"- Average episode length: {strategy_data['avg_episode_length'].mean():.1f} ± {strategy_data['avg_episode_length'].std():.1f}\n"
            report += f"- Training stability (std): {strategy_data['std_training_reward'].mean():.3f}\n\n"
        
        return report
