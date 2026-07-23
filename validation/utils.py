"""
Shared utilities for CRAFT validation pipeline.

Functions:
    scaffold_split         Bemis-Murcko scaffold split (train/val/test)
    rf_gb_ensemble         RF+GB ensemble regressor or classifier
    evaluate_regression    R2, MAE, RMSE metrics
    evaluate_classification AUC-ROC, AUPRC, F1, accuracy metrics
    multi_seed_run         Run experiment with five seeds, collect stats
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy import stats
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
)
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ---------------------------------------------------------------------------
# Scaffold splitting (Bemis-Murcko generic frameworks)
# ---------------------------------------------------------------------------

def _get_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core else smiles


def scaffold_split(
    smiles_list: list[str],
    labels: np.ndarray,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by Bemis-Murcko scaffold. Returns train/val/test indices.

    Parameters
    ----------
    smiles_list : list[str]
    labels      : np.ndarray, shape (N,) - used only for stratification checks
    train_frac  : float
    val_frac    : float
    seed        : int

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
    """
    rng = np.random.default_rng(seed)

    # Group indices by scaffold
    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaffold_to_indices[_get_scaffold(smi)].append(i)

    # Shuffle scaffold groups
    groups = list(scaffold_to_indices.values())
    rng.shuffle(groups)

    n = len(smiles_list)
    train_target = int(n * train_frac)
    val_target   = int(n * val_frac)

    train_idx, val_idx, test_idx = [], [], []
    for group in groups:
        if len(train_idx) < train_target:
            train_idx.extend(group)
        elif len(val_idx) < val_target:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return (
        np.array(train_idx, dtype=np.intp),
        np.array(val_idx,   dtype=np.intp),
        np.array(test_idx,  dtype=np.intp),
    )


# ---------------------------------------------------------------------------
# RF+GB ensemble
# ---------------------------------------------------------------------------

def rf_gb_regressor(n_jobs: int = -1, seed: int = 42):
    """500-tree RF + 200-tree GB regressor ensemble (matches paper spec)."""
    rf = RandomForestRegressor(n_estimators=500, n_jobs=n_jobs, random_state=seed)
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                    learning_rate=0.1, random_state=seed)
    return rf, gb


def rf_gb_classifier(n_jobs: int = -1, seed: int = 42):
    """500-tree RF + 200-tree GB classifier ensemble (matches paper spec)."""
    rf = RandomForestClassifier(n_estimators=500, n_jobs=n_jobs,
                                 random_state=seed, class_weight="balanced")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                     learning_rate=0.1, random_state=seed)
    return rf, gb


def ensemble_predict_proba(rf, gb, X_test: np.ndarray) -> np.ndarray:
    """Average RF and GB predicted probabilities."""
    p_rf = rf.predict_proba(X_test)[:, 1]
    p_gb = gb.predict_proba(X_test)[:, 1]
    return (p_rf + p_gb) / 2


def ensemble_predict_regression(rf, gb, X_test: np.ndarray) -> np.ndarray:
    """Average RF and GB predicted values."""
    return (rf.predict(X_test) + gb.predict(X_test)) / 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2":   float(r2_score(y_true, y_pred)),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate_classification(y_true: np.ndarray, y_prob: np.ndarray,
                             threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc_roc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc_roc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float("nan")
    return {
        "auc_roc":  auc_roc,
        "auprc":    auprc,
        "f1":       float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456, 789, 2024]


def multi_seed_run(
    experiment_fn: Callable[[int], dict[str, float]],
    seeds: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Run experiment_fn(seed) for each seed and return
    {metric: {mean, std, ci_low, ci_high}}.

    Parameters
    ----------
    experiment_fn : callable(seed: int) -> dict[metric_name: float]
    seeds         : list of int seeds (default: [42, 123, 456, 789, 2024])

    Returns
    -------
    summary : dict  e.g. {"r2": {"mean": 0.54, "std": 0.04, ...}, ...}
    """
    if seeds is None:
        seeds = SEEDS

    all_results: dict[str, list[float]] = defaultdict(list)
    for seed in seeds:
        result = experiment_fn(seed)
        for metric, value in result.items():
            all_results[metric].append(value)

    summary = {}
    for metric, values in all_results.items():
        arr = np.array(values)
        n = len(arr)
        sem = arr.std() / np.sqrt(n)
        ci_lo = arr.mean() - 1.96 * sem
        ci_hi = arr.mean() + 1.96 * sem
        summary[metric] = {
            "mean":    float(arr.mean()),
            "std":     float(arr.std()),
            "ci_low":  float(ci_lo),
            "ci_high": float(ci_hi),
            "values":  [float(v) for v in values],
        }

    return summary


def wilcoxon_paired(a_values: list[float], b_values: list[float]) -> float:
    """One-sided Wilcoxon signed-rank test for H1: b > a. Returns p-value."""
    stat, p = stats.wilcoxon(a_values, b_values, alternative="less")
    return float(p)
