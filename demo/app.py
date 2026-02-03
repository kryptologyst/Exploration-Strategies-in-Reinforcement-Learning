"""Streamlit demo for exploration strategies comparison."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from algorithms.q_learning import DQNAgent, QLearningAgent
from evaluation.trainer import Evaluator, Trainer, TrainingConfig
from utils.utils import set_seed


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Exploration Strategies in RL",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Exploration Strategies in Reinforcement Learning")
    st.markdown("""
    This demo compares different exploration strategies in reinforcement learning:
    - **Epsilon-Greedy**: Random exploration with decaying probability
    - **UCB**: Upper Confidence Bound for optimistic exploration
    - **Thompson Sampling**: Bayesian exploration using posterior sampling
    - **Curiosity-Driven**: Intrinsic motivation based on prediction error
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Agent type selection
    agent_type = st.sidebar.selectbox(
        "Agent Type",
        ["tabular", "dqn"],
        help="Tabular Q-learning vs Deep Q-Network"
    )
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"],
        help="RL environment to use"
    )
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    num_episodes = st.sidebar.slider("Number of Episodes", 100, 2000, 1000)
    num_runs = st.sidebar.slider("Number of Runs", 1, 10, 3)
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=0)
    
    # Strategy selection
    st.sidebar.subheader("Strategies to Compare")
    strategies = []
    if st.sidebar.checkbox("Epsilon-Greedy", value=True):
        strategies.append("epsilon_greedy")
    if st.sidebar.checkbox("UCB", value=True):
        strategies.append("ucb")
    if st.sidebar.checkbox("Thompson Sampling", value=True):
        strategies.append("thompson")
    if st.sidebar.checkbox("Curiosity-Driven", value=True):
        strategies.append("curiosity")
    
    if not strategies:
        st.error("Please select at least one strategy to compare!")
        return
    
    # Run comparison button
    if st.sidebar.button("🚀 Run Comparison", type="primary"):
        run_comparison(agent_type, env_name, strategies, num_episodes, num_runs, seed)


def run_comparison(agent_type, env_name, strategies, num_episodes, num_runs, seed):
    """Run strategy comparison and display results."""
    
    # Set random seed
    set_seed(seed)
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create evaluation configuration
    config = TrainingConfig(
        env_name=env_name,
        num_episodes=num_episodes,
        seed=seed,
        eval_frequency=max(1, num_episodes // 10),
        num_eval_episodes=5
    )
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Initialize results storage
    all_results = []
    training_curves = {}
    
    total_experiments = len(strategies) * num_runs
    current_experiment = 0
    
    # Run experiments
    for strategy in strategies:
        st.subheader(f"Testing {strategy.replace('_', ' ').title()}")
        
        strategy_results = []
        strategy_curves = []
        
        for run in range(num_runs):
            current_experiment += 1
            progress = current_experiment / total_experiments
            progress_bar.progress(progress)
            status_text.text(f"Running {strategy} - Run {run + 1}/{num_runs}")
            
            # Create agent
            if agent_type == "tabular":
                agent = QLearningAgent(
                    action_space_size=get_action_space_size(env_name),
                    exploration_strategy=strategy,
                    seed=seed + run
                )
            else:
                agent = DQNAgent(
                    state_dim=get_state_dim(env_name),
                    action_space_size=get_action_space_size(env_name),
                    exploration_strategy=strategy,
                    seed=seed + run
                )
            
            # Train agent
            trainer = Trainer(agent, config, f"temp_outputs/{strategy}_run_{run}")
            stats = trainer.train()
            
            # Store results
            final_eval_reward = stats["eval_rewards"][-1] if stats["eval_rewards"] else 0
            final_eval_std = stats["eval_std"][-1] if stats["eval_std"] else 0
            
            strategy_results.append({
                "strategy": strategy,
                "run": run,
                "final_eval_reward": final_eval_reward,
                "final_eval_std": final_eval_std,
                "avg_training_reward": np.mean(stats["episode_rewards"]),
                "std_training_reward": np.std(stats["episode_rewards"]),
                "max_training_reward": np.max(stats["episode_rewards"]),
                "avg_episode_length": np.mean(stats["episode_lengths"]),
            })
            
            # Store training curves
            strategy_curves.append({
                "episodes": list(range(len(stats["episode_rewards"]))),
                "rewards": stats["episode_rewards"],
                "eval_rewards": stats["eval_rewards"],
                "eval_episodes": list(range(0, len(stats["eval_rewards"]) * config.eval_frequency, config.eval_frequency))
            })
        
        all_results.extend(strategy_results)
        training_curves[strategy] = strategy_curves
    
    progress_bar.progress(1.0)
    status_text.text("Analysis complete!")
    
    # Display results
    display_results(all_results, training_curves, strategies)


def display_results(all_results, training_curves, strategies):
    """Display comparison results."""
    
    results_df = pd.DataFrame(all_results)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    summary = results_df.groupby("strategy").agg({
        "final_eval_reward": ["mean", "std"],
        "avg_training_reward": ["mean", "std"],
        "avg_episode_length": ["mean", "std"],
    }).round(3)
    
    st.dataframe(summary)
    
    # Best strategy
    best_strategy = results_df.groupby("strategy")["final_eval_reward"].mean().idxmax()
    best_reward = results_df.groupby("strategy")["final_eval_reward"].mean().max()
    
    st.success(f"🏆 Best performing strategy: **{best_strategy.replace('_', ' ').title()}** "
              f"(Average reward: {best_reward:.2f})")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Training Curves", "📊 Comparison Charts", "📋 Detailed Results", "🔍 Analysis"])
    
    with tab1:
        plot_training_curves(training_curves, strategies)
    
    with tab2:
        plot_comparison_charts(results_df)
    
    with tab3:
        display_detailed_results(results_df)
    
    with tab4:
        display_analysis(results_df)


def plot_training_curves(training_curves, strategies):
    """Plot training curves for all strategies."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
    
    for i, strategy in enumerate(strategies):
        color = colors[i]
        
        # Plot episode rewards
        for run_data in training_curves[strategy]:
            axes[0, 0].plot(
                run_data["episodes"], 
                run_data["rewards"], 
                alpha=0.3, 
                color=color
            )
        
        # Plot average episode rewards
        all_episodes = training_curves[strategy][0]["episodes"]
        avg_rewards = np.mean([run_data["rewards"] for run_data in training_curves[strategy]], axis=0)
        axes[0, 0].plot(all_episodes, avg_rewards, color=color, linewidth=2, label=strategy.replace('_', ' ').title())
        
        # Plot evaluation rewards
        for run_data in training_curves[strategy]:
            if run_data["eval_rewards"]:
                axes[0, 1].plot(
                    run_data["eval_episodes"], 
                    run_data["eval_rewards"], 
                    alpha=0.3, 
                    color=color
                )
        
        # Plot average evaluation rewards
        if training_curves[strategy][0]["eval_rewards"]:
            avg_eval_rewards = np.mean([run_data["eval_rewards"] for run_data in training_curves[strategy]], axis=0)
            axes[0, 1].plot(training_curves[strategy][0]["eval_episodes"], avg_eval_rewards, 
                           color=color, linewidth=2, label=strategy.replace('_', ' ').title())
    
    axes[0, 0].set_title("Episode Rewards")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].set_title("Evaluation Rewards")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot reward distributions
    strategy_rewards = {}
    for strategy in strategies:
        strategy_rewards[strategy] = []
        for run_data in training_curves[strategy]:
            strategy_rewards[strategy].extend(run_data["rewards"])
    
    axes[1, 0].hist([strategy_rewards[s] for s in strategies], 
                   bins=30, alpha=0.7, label=[s.replace('_', ' ').title() for s in strategies])
    axes[1, 0].set_title("Reward Distribution")
    axes[1, 0].set_xlabel("Reward")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot learning progress (moving average)
    window_size = 50
    for i, strategy in enumerate(strategies):
        color = colors[i]
        for run_data in training_curves[strategy]:
            moving_avg = np.convolve(run_data["rewards"], np.ones(window_size)/window_size, mode='valid')
            axes[1, 1].plot(range(window_size-1, len(run_data["rewards"])), moving_avg, 
                           alpha=0.3, color=color)
        
        # Plot average moving average
        all_moving_avgs = []
        for run_data in training_curves[strategy]:
            moving_avg = np.convolve(run_data["rewards"], np.ones(window_size)/window_size, mode='valid')
            all_moving_avgs.append(moving_avg)
        
        if all_moving_avgs:
            avg_moving_avg = np.mean(all_moving_avgs, axis=0)
            axes[1, 1].plot(range(window_size-1, len(training_curves[strategy][0]["rewards"])), 
                           avg_moving_avg, color=color, linewidth=2, 
                           label=strategy.replace('_', ' ').title())
    
    axes[1, 1].set_title(f"Learning Progress (Moving Average, window={window_size})")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Reward")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)


def plot_comparison_charts(results_df):
    """Plot comparison charts."""
    
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
    
    # Training stability
    sns.boxplot(data=results_df, x="strategy", y="std_training_reward", ax=axes[1, 1])
    axes[1, 1].set_title("Training Stability (Reward Std)")
    axes[1, 1].set_ylabel("Standard Deviation")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    st.pyplot(fig)


def display_detailed_results(results_df):
    """Display detailed results table."""
    
    st.subheader("📋 Detailed Results")
    
    # Format strategy names
    display_df = results_df.copy()
    display_df["strategy"] = display_df["strategy"].str.replace("_", " ").str.title()
    
    # Round numerical columns
    numerical_cols = ["final_eval_reward", "final_eval_std", "avg_training_reward", 
                      "std_training_reward", "max_training_reward", "avg_episode_length"]
    for col in numerical_cols:
        display_df[col] = display_df[col].round(3)
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="exploration_strategies_results.csv",
        mime="text/csv"
    )


def display_analysis(results_df):
    """Display analysis and insights."""
    
    st.subheader("🔍 Analysis and Insights")
    
    # Statistical significance test
    from scipy import stats
    
    strategies = results_df["strategy"].unique()
    
    st.write("**Statistical Significance Tests:**")
    
    for i, strategy1 in enumerate(strategies):
        for strategy2 in strategies[i+1:]:
            data1 = results_df[results_df["strategy"] == strategy1]["final_eval_reward"]
            data2 = results_df[results_df["strategy"] == strategy2]["final_eval_reward"]
            
            t_stat, p_value = stats.ttest_ind(data1, data2)
            
            significance = "significant" if p_value < 0.05 else "not significant"
            st.write(f"- {strategy1.replace('_', ' ').title()} vs {strategy2.replace('_', ' ').title()}: "
                    f"p-value = {p_value:.4f} ({significance})")
    
    # Key insights
    st.write("**Key Insights:**")
    
    best_strategy = results_df.groupby("strategy")["final_eval_reward"].mean().idxmax()
    worst_strategy = results_df.groupby("strategy")["final_eval_reward"].mean().idxmin()
    
    st.write(f"- **Best Strategy**: {best_strategy.replace('_', ' ').title()}")
    st.write(f"- **Worst Strategy**: {worst_strategy.replace('_', ' ').title()}")
    
    # Stability analysis
    stability_df = results_df.groupby("strategy")["std_training_reward"].mean().sort_values()
    st.write("- **Most Stable**:", stability_df.index[0].replace('_', ' ').title())
    st.write("- **Least Stable**:", stability_df.index[-1].replace('_', ' ').title())
    
    # Sample efficiency
    efficiency_df = results_df.groupby("strategy")["avg_episode_length"].mean().sort_values()
    st.write("- **Most Sample Efficient**:", efficiency_df.index[0].replace('_', ' ').title())
    st.write("- **Least Sample Efficient**:", efficiency_df.index[-1].replace('_', ' ').title())


def get_action_space_size(env_name):
    """Get action space size for environment."""
    action_sizes = {
        "CartPole-v1": 2,
        "MountainCar-v0": 3,
        "Acrobot-v1": 3,
    }
    return action_sizes.get(env_name, 2)


def get_state_dim(env_name):
    """Get state dimension for environment."""
    state_dims = {
        "CartPole-v1": 4,
        "MountainCar-v0": 2,
        "Acrobot-v1": 6,
    }
    return state_dims.get(env_name, 4)


if __name__ == "__main__":
    main()
