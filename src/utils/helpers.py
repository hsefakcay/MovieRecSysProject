"""
Common utility functions shared across the project.

Includes:
  - Reproducibility seed setter
  - Config loader (YAML)
  - Result saver / loader
  - Pretty-print helpers
"""

import os
import json
import random
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    Call this at the top of every notebook and training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # MPS does not have a manual_seed method in older PyTorch versions
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(config_path: str = "experiments/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_results(results: dict, path: str) -> None:
    """Save experiment results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {path}")


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path, "r") as f:
        return json.load(f)


def print_metrics_table(df: pd.DataFrame, model_name: str = "") -> None:
    """Pretty-print an evaluation metrics DataFrame."""
    print(f"\n{'='*55}")
    if model_name:
        print(f"  {model_name}")
    print(f"{'='*55}")
    print(df.to_string(float_format="{:.4f}".format))
    print(f"{'='*55}\n")


def compare_models(results: dict[str, pd.DataFrame], k: int = 10) -> pd.DataFrame:
    """
    Build a comparison table for K=k across multiple models.

    Args:
        results: {model_name: metrics_df (output of evaluate_ranking)}
        k: the K value to compare

    Returns:
        pd.DataFrame with one row per model
    """
    rows = []
    for name, df in results.items():
        if k in df.index:
            row = df.loc[k].to_dict()
            row["Model"] = name
            rows.append(row)
    comp = pd.DataFrame(rows).set_index("Model")
    return comp
