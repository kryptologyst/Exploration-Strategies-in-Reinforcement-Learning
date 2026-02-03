"""Utility functions for the exploration strategies project."""

import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device.
    
    Returns:
        PyTorch device (CUDA, MPS, or CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_env(env_name: str, seed: Optional[int] = None) -> gym.Env:
    """Create and configure environment.
    
    Args:
        env_name: Name of the environment
        seed: Random seed for environment
        
    Returns:
        Configured environment
    """
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_env_info(env: gym.Env) -> Dict[str, Any]:
    """Get environment information.
    
    Args:
        env: Gymnasium environment
        
    Returns:
        Dictionary with environment information
    """
    return {
        "action_space": env.action_space,
        "observation_space": env.observation_space,
        "action_space_size": env.action_space.n if hasattr(env.action_space, 'n') else None,
        "observation_space_size": env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None,
        "max_episode_steps": getattr(env.spec, 'max_episode_steps', None),
    }


def normalize_rewards(rewards: List[float], method: str = "z_score") -> List[float]:
    """Normalize rewards for better training stability.
    
    Args:
        rewards: List of rewards
        method: Normalization method ("z_score", "min_max", "robust")
        
    Returns:
        Normalized rewards
    """
    rewards_array = np.array(rewards)
    
    if method == "z_score":
        mean_reward = np.mean(rewards_array)
        std_reward = np.std(rewards_array)
        if std_reward > 0:
            return ((rewards_array - mean_reward) / std_reward).tolist()
        else:
            return rewards
    
    elif method == "min_max":
        min_reward = np.min(rewards_array)
        max_reward = np.max(rewards_array)
        if max_reward > min_reward:
            return ((rewards_array - min_reward) / (max_reward - min_reward)).tolist()
        else:
            return rewards
    
    elif method == "robust":
        median_reward = np.median(rewards_array)
        mad = np.median(np.abs(rewards_array - median_reward))
        if mad > 0:
            return ((rewards_array - median_reward) / mad).tolist()
        else:
            return rewards
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def calculate_confidence_interval(
    data: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate confidence interval for data.
    
    Args:
        data: List of values
        confidence: Confidence level (0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data_array = np.array(data)
    mean = np.mean(data_array)
    std = np.std(data_array)
    n = len(data_array)
    
    # Calculate standard error
    se = std / np.sqrt(n)
    
    # Calculate critical value (approximate for normal distribution)
    alpha = 1 - confidence
    z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    
    margin_error = z_score * se
    
    return mean, mean - margin_error, mean + margin_error


def moving_average(data: List[float], window_size: int) -> List[float]:
    """Calculate moving average of data.
    
    Args:
        data: List of values
        window_size: Size of moving window
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i + 1]
        moving_avg.append(np.mean(window_data))
    
    return moving_avg


def create_output_directory(base_dir: str, experiment_name: str) -> str:
    """Create output directory for experiment.
    
    Args:
        base_dir: Base directory path
        experiment_name: Name of the experiment
        
    Returns:
        Path to created directory
    """
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_experiment_results(
    results: Dict[str, Any], 
    output_dir: str, 
    filename: str = "results.json"
) -> None:
    """Save experiment results to file.
    
    Args:
        results: Dictionary with experiment results
        output_dir: Output directory path
        filename: Name of the results file
    """
    import json
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_experiment_results(
    filepath: str
) -> Dict[str, Any]:
    """Load experiment results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Dictionary with experiment results
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_progress_bar(
    current: int, 
    total: int, 
    prefix: str = "Progress", 
    suffix: str = "Complete",
    length: int = 50
) -> None:
    """Print a progress bar.
    
    Args:
        current: Current progress
        total: Total progress
        prefix: Prefix text
        suffix: Suffix text
        length: Length of progress bar
    """
    percent = 100 * (current / float(total))
    filled_length = int(length * current // total)
    bar = "█" * filled_length + "-" * (length - filled_length)
    print(f"\r{prefix} |{bar}| {percent:.1f}% {suffix}", end="", flush=True)
    
    if current == total:
        print()


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ["env_name", "num_episodes", "learning_rate"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    if config["num_episodes"] <= 0:
        raise ValueError("num_episodes must be positive")
    
    if not 0 < config["learning_rate"] <= 1:
        raise ValueError("learning_rate must be between 0 and 1")
    
    return True


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage.
    
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / 1024 / 1024,  # MB
        "vms": memory_info.vms / 1024 / 1024,  # MB
        "percent": process.memory_percent(),
    }


def log_system_info() -> Dict[str, Any]:
    """Log system information for reproducibility.
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "gymnasium_version": gym.__version__,
    }
