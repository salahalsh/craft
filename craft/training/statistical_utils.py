"""
Multi-seed statistical utilities for CRAFT validation.

Provides:
- SEEDS: canonical 5 seeds for reproducibility
- aggregate_metrics(): mean, std, 95% CI from per-seed results
- paired_wilcoxon_test(): significance test between conditions
- bootstrap_ci(): bootstrap confidence intervals for small samples
- format_metric_with_ci(): formatted string for paper tables
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Canonical seeds - used across all validation tasks for reproducibility.
# 5 seeds required: minimum for Wilcoxon signed-rank p_min = 0.0625 < 0.05
# (with n=3 seeds the minimum possible p-value is 0.125, which cannot reach α=0.05)
SEEDS = [42, 123, 456, 789, 2024]


def aggregate_metrics(per_seed_values: List[float]) -> Dict:
    """
    Aggregate per-seed metric values into summary statistics.

    Args:
        per_seed_values: List of metric values, one per seed

    Returns:
        Dict with keys: mean, std, ci95_low, ci95_high, n_seeds, per_seed
    """
    values = np.array(per_seed_values, dtype=np.float64)
    n = len(values)

    if n == 0:
        return {
            'mean': float('nan'),
            'std': float('nan'),
            'ci95_low': float('nan'),
            'ci95_high': float('nan'),
            'n_seeds': 0,
            'per_seed': [],
        }

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    # 95% CI via bootstrap (more robust for n=5 than t-distribution)
    ci_low, ci_high = bootstrap_ci(per_seed_values)

    return {
        'mean': mean,
        'std': std,
        'ci95_low': ci_low,
        'ci95_high': ci_high,
        'n_seeds': n,
        'per_seed': [float(v) for v in values],
    }


def paired_wilcoxon_test(baseline_values: List[float],
                         augmented_values: List[float]) -> Dict:
    """
    Wilcoxon signed-rank test comparing two conditions across seeds.

    Tests whether augmented_values are significantly greater than baseline_values.

    Args:
        baseline_values: Metric values for baseline condition (one per seed)
        augmented_values: Metric values for augmented condition (one per seed)

    Returns:
        Dict with statistic, p_value, significant, n_seeds, mean_delta
    """
    from scipy.stats import wilcoxon

    baseline = np.array(baseline_values, dtype=np.float64)
    augmented = np.array(augmented_values, dtype=np.float64)
    differences = augmented - baseline

    n = len(differences)
    mean_delta = float(np.mean(differences))

    if n < 5:
        logger.warning("Wilcoxon test with n=%d seeds - low power", n)

    # If all differences are zero, test is undefined
    if np.all(differences == 0):
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'n_seeds': n,
            'mean_delta': 0.0,
        }

    try:
        stat, p_value = wilcoxon(
            augmented, baseline, alternative='greater', zero_method='wilcox')
    except ValueError:
        # All differences identical or other edge case
        stat, p_value = 0.0, 1.0

    return {
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'n_seeds': n,
        'mean_delta': mean_delta,
    }


def bootstrap_ci(values: List[float], confidence: float = 0.95,
                 n_bootstrap: int = 10000, seed: int = 42) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for small samples.

    Args:
        values: Observed metric values
        confidence: Confidence level (default 0.95)
        n_bootstrap: Number of bootstrap resamples
        seed: Random seed for reproducibility

    Returns:
        (ci_low, ci_high) tuple
    """
    arr = np.array(values, dtype=np.float64)
    n = len(arr)

    if n <= 1:
        return (float(arr[0]) if n == 1 else float('nan'),
                float(arr[0]) if n == 1 else float('nan'))

    rng = np.random.RandomState(seed)
    bootstrap_means = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - confidence
    ci_low = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_high = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return ci_low, ci_high


def format_metric_with_ci(mean: float, ci_low: float, ci_high: float,
                          fmt: str = '.3f') -> str:
    """
    Format metric as 'mean [ci_low, ci_high]' for paper tables.

    Args:
        mean: Mean value
        ci_low: Lower CI bound
        ci_high: Upper CI bound
        fmt: Format specifier (default '.3f')

    Returns:
        Formatted string like '0.579 [0.561, 0.598]'
    """
    return f"{mean:{fmt}} [{ci_low:{fmt}}, {ci_high:{fmt}}]"


def aggregate_task_metrics(per_seed_results: List[Dict],
                           metric_paths: List[Tuple[str, ...]]) -> Dict:
    """
    Aggregate specific metric paths from per-seed result dicts.

    Args:
        per_seed_results: List of full result dicts, one per seed
        metric_paths: List of key paths to aggregate, e.g.
                      [('baseline_ecfp', 'test', 'r2'),
                       ('craft_augmented', 'test', 'r2')]

    Returns:
        Dict mapping '.'.join(path) to aggregate_metrics() output
    """
    aggregated = {}

    for path in metric_paths:
        values = []
        for result in per_seed_results:
            # Navigate the nested dict
            val = result
            try:
                for key in path:
                    val = val[key]
                values.append(float(val))
            except (KeyError, TypeError, ValueError):
                logger.warning("Missing metric path %s in seed result", path)

        path_key = '.'.join(path)
        aggregated[path_key] = aggregate_metrics(values)

    return aggregated
