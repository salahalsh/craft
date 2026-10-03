"""
CRAFT Validation Tasks - 4 tasks demonstrating CRAFT improves ML predictions.

Task 1: BindingAffinityTask    - Multi-target pIC50 regression
Task 2: ADMETTask              - Pooled Tox21 classification
Task 3: SelectivityTask        - Fingerprint similarity analysis (no ML)
Task 4: PolypharmacologyTask   - Cross-target binary classification

All ML tasks follow the POLY-X pattern: sklearn RF(500)+GB(200), scaffold
split (80/10/10, seed=42), metrics saved as JSON.
"""

import gc
import json
import logging
import os
import psutil
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, accuracy_score, f1_score,
                             average_precision_score)

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Base class
# ───────────────────────────────────────────────────────────────────────────

class BaseValidationTask:
    """Shared infrastructure for all CRAFT validation tasks."""

    MEMORY_LIMIT_PERCENT = 95  # Max system memory usage before pausing/aborting

    def __init__(self, output_dir: str, max_targets: int = 0,
                 rf_n_estimators: int = 500, gb_n_estimators: int = 200,
                 random_seed: int = 42,
                 n_jobs: int = -1):
        self.output_dir = output_dir
        self.max_targets = max_targets
        self.rf_n_estimators = rf_n_estimators
        self.gb_n_estimators = gb_n_estimators
        self.seed = random_seed
        self.n_jobs = n_jobs  # -1 = all cores, positive int = exact count
        os.makedirs(output_dir, exist_ok=True)

    def run(self) -> Dict:
        raise NotImplementedError

    def _check_memory(self, context: str = '') -> bool:
        """Check system memory usage. Returns True if OK, False if over limit.
        Attempts gc.collect() if above 70% to recover before hitting limit."""
        mem = psutil.virtual_memory()
        pct = mem.percent
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        if pct >= self.MEMORY_LIMIT_PERCENT:
            logger.warning(
                "MEMORY LIMIT: %.1f%% used (%.1fGB/%.1fGB) %s - "
                "running gc.collect()...", pct, used_gb, total_gb, context)
            gc.collect()
            mem = psutil.virtual_memory()
            pct = mem.percent
            if pct >= self.MEMORY_LIMIT_PERCENT:
                logger.error(
                    "MEMORY STILL HIGH after gc: %.1f%% %s", pct, context)
                return False
            logger.info("Memory recovered to %.1f%% after gc", pct)
        elif pct >= 80:
            gc.collect()
            logger.info("Memory at %.1f%% %s - proactive gc done", pct, context)
        return True

    def _log_memory(self, label: str = ''):
        """Log current memory usage."""
        mem = psutil.virtual_memory()
        proc = psutil.Process()
        proc_mb = proc.memory_info().rss / (1024 ** 2)
        logger.info("Memory [%s]: system=%.1f%% (%.1fGB), process=%.0fMB",
                    label, mem.percent, mem.used / (1024 ** 3), proc_mb)

    def run_multi_seed(self, seeds: Optional[List[int]] = None) -> Dict:
        """
        Run the task with multiple random seeds and aggregate results.

        Args:
            seeds: List of random seeds (default: SEEDS from statistical_utils)

        Returns:
            Dict with 'per_seed' results and 'aggregated' summary with CI
        """
        from .statistical_utils import SEEDS as DEFAULT_SEEDS

        if seeds is None:
            seeds = DEFAULT_SEEDS

        # Checkpoint directory for per-seed results
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)

        per_seed_results = []
        for seed in seeds:
            checkpoint_file = os.path.join(
                checkpoint_dir, f'seed_{seed}.json')

            # Resume: load completed seed from checkpoint
            if os.path.exists(checkpoint_file):
                logger.info("Resuming seed %d (%d/%d) from checkpoint",
                            seed, seeds.index(seed) + 1, len(seeds))
                with open(checkpoint_file) as f:
                    result = json.load(f)
                per_seed_results.append(result)
                continue

            # Memory check before starting a new seed
            if not self._check_memory(f'before seed {seed}'):
                logger.error(
                    "ABORTING multi-seed at seed %d: memory too high. "
                    "Completed seeds are checkpointed - resume after restart.",
                    seed)
                break

            self._log_memory(f'seed {seed} start ({seeds.index(seed)+1}/{len(seeds)})')
            logger.info("Running seed %d (%d/%d)...", seed,
                        seeds.index(seed) + 1, len(seeds))
            self.seed = seed
            result = self.run()

            # Checkpoint logic: LOFO requires all families; other tasks always checkpoint
            is_lofo = 'n_families_tested' in result or 'per_family' in result
            if is_lofo:
                n_families = result.get('n_families_tested', 0)
                per_family = result.get('per_family', {})
                if n_families >= 5 or len(per_family) >= 5:
                    per_seed_results.append(result)
                    with open(checkpoint_file, 'w') as f:
                        json.dump(result, f, indent=2, default=str)
                    logger.info("Checkpoint saved: seed %d (%d families) → %s",
                                seed, n_families, checkpoint_file)
                else:
                    logger.warning(
                        "Seed %d incomplete (%d families) - NOT checkpointed. "
                        "Fold checkpoints preserved for resume.",
                        seed, n_families)
                    break  # If one seed can't complete, next won't either
            else:
                # Non-LOFO tasks: always checkpoint completed seeds
                per_seed_results.append(result)
                with open(checkpoint_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                logger.info("Checkpoint saved: seed %d → %s",
                            seed, checkpoint_file)

            # Force cleanup between seeds
            gc.collect()
            self._log_memory(f'seed {seed} done')

        # Aggregate - subclasses implement _aggregate_results()
        aggregated = self._aggregate_results(per_seed_results, seeds)

        multi_seed_output = {
            'multi_seed': True,
            'seeds': seeds,
            'n_seeds': len(seeds),
            'per_seed': {str(s): r for s, r in zip(seeds, per_seed_results)},
            'aggregated': aggregated,
        }

        self._save_results(multi_seed_output, {
            'task': self.__class__.__name__,
            'multi_seed': True,
            'seeds': seeds,
            'n_seeds': len(seeds),
        }, f'{self.__class__.__name__}_multi_seed')

        return multi_seed_output

    def _aggregate_results(self, per_seed_results: List[Dict],
                           seeds: List[int]) -> Dict:
        """Override in subclasses to aggregate per-seed metrics."""
        return {'note': 'No aggregation implemented for this task'}

    def _train_rf_gb_regressor(self, X_train, y_train):
        """Train RF+GB regressor ensemble. Returns (rf, gb)."""
        rf = RandomForestRegressor(
            n_estimators=self.rf_n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        gb = GradientBoostingRegressor(
            n_estimators=self.gb_n_estimators,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=self.seed,
        )
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        return rf, gb

    def _train_rf_gb_classifier(self, X_train, y_train):
        """Train RF+GB classifier ensemble with balanced weights."""
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=None,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=self.n_jobs,
            random_state=self.seed,
        )
        gb = GradientBoostingClassifier(
            n_estimators=self.gb_n_estimators,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=self.seed,
        )
        rf.fit(X_train, y_train)
        gb.fit(X_train, y_train)
        return rf, gb

    def _ensemble_predict_regression(self, rf, gb, X):
        """Average of RF and GB predictions."""
        return (rf.predict(X) + gb.predict(X)) / 2

    def _ensemble_predict_proba(self, rf, gb, X):
        """Average of RF and GB predicted probabilities."""
        return (rf.predict_proba(X)[:, 1] + gb.predict_proba(X)[:, 1]) / 2

    def _regression_metrics(self, y_true, y_pred) -> Dict:
        """Compute regression metrics."""
        return {
            'r2': float(r2_score(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        }

    def _classification_metrics(self, y_true, y_proba) -> Dict:
        """Compute classification metrics."""
        y_pred = (y_proba >= 0.5).astype(int)
        m = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        }
        try:
            m['auc_roc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            m['auc_roc'] = float('nan')
        try:
            m['auprc'] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            m['auprc'] = float('nan')
        return m

    def _scaffold_split(self, smiles_list: List[str]
                        ) -> Tuple[List[int], List[int], List[int]]:
        """
        Scaffold split on SMILES. Returns (train_idx, val_idx, test_idx).

        80/10/10 split using Murcko scaffolds, seed=42.
        """
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffolds = {}
        for idx, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False)
                else:
                    scaffold = f'_invalid_{idx}'
            except Exception:
                scaffold = f'_error_{idx}'

            if scaffold not in scaffolds:
                scaffolds[scaffold] = []
            scaffolds[scaffold].append(idx)

        # Sort by descending scaffold size, then shuffle for randomized assignment
        scaffold_sets = sorted(
            scaffolds.values(), key=lambda g: (-len(g), min(g)))

        rng = random.Random(self.seed)
        rng.shuffle(scaffold_sets)

        n_total = len(smiles_list)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        n_test = n_total - n_train - n_val

        # Two-phase assignment to handle datasets with very large scaffold
        # groups (e.g., Tox21 has groups of 1775 and 1474 compounds, each
        # larger than n_val=782 or n_test=783). Such groups MUST go to train
        # to avoid overshooting the smaller bins.
        min_bin = min(n_val, n_test)
        train_idx = []
        small_groups = []
        for group in scaffold_sets:
            if len(group) > min_bin:
                train_idx.extend(group)
            else:
                small_groups.append(group)

        # Greedy assignment for remaining groups (all fit in any bin)
        val_idx, test_idx = [], []
        for group in small_groups:
            if len(train_idx) < n_train:
                train_idx.extend(group)
            elif len(val_idx) < n_val:
                val_idx.extend(group)
            else:
                test_idx.extend(group)

        logger.info("Scaffold split: train=%d, val=%d, test=%d",
                     len(train_idx), len(val_idx), len(test_idx))
        return train_idx, val_idx, test_idx

    def _compound_level_scaffold_split(self, smiles_list: List[str],
                                        compound_to_rows: Dict[str, List[int]]
                                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scaffold split at compound level, then map back to row indices.

        Ensures same compound is in the same split across all tasks.
        Returns boolean masks: (train_mask, val_mask, test_mask).
        """
        unique_smiles = sorted(compound_to_rows.keys())
        train_comp, val_comp, test_comp = self._scaffold_split(unique_smiles)

        train_set = set(unique_smiles[i] for i in train_comp)
        val_set = set(unique_smiles[i] for i in val_comp)
        test_set = set(unique_smiles[i] for i in test_comp)

        n_rows = len(smiles_list)
        train_mask = np.zeros(n_rows, dtype=bool)
        val_mask = np.zeros(n_rows, dtype=bool)
        test_mask = np.zeros(n_rows, dtype=bool)

        for smi, rows in compound_to_rows.items():
            if smi in train_set:
                for r in rows:
                    train_mask[r] = True
            elif smi in val_set:
                for r in rows:
                    val_mask[r] = True
            elif smi in test_set:
                for r in rows:
                    test_mask[r] = True

        logger.info("Compound-level scaffold split: train=%d, val=%d, test=%d rows",
                     train_mask.sum(), val_mask.sum(), test_mask.sum())
        return train_mask, val_mask, test_mask

    def _save_results(self, metrics: Dict, config: Dict, task_name: str):
        """Save metrics.json and config.json to output directory."""
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        config_path = os.path.join(self.output_dir, 'config.json')

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

        logger.info("Saved %s results to %s", task_name, self.output_dir)

    def _save_predictions(self, predictions: Dict, suffix: str = ''):
        """Save raw y_true/y_pred arrays for figure generation.

        predictions: dict of {label: {'y_true': array, 'y_pred': array, ...}}
        suffix: e.g. 'seed_42' to distinguish multi-seed runs
        """
        pred_dir = os.path.join(self.output_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        fname = f'predictions_{suffix}.npz' if suffix else 'predictions.npz'
        save_dict = {}
        for label, arrays in predictions.items():
            for key, arr in arrays.items():
                save_dict[f'{label}__{key}'] = np.array(arr)
        np.savez_compressed(os.path.join(pred_dir, fname), **save_dict)
        logger.info("Saved predictions to %s/%s", pred_dir, fname)


# ───────────────────────────────────────────────────────────────────────────
# Task 1: Multi-Target Binding Affinity (Regression)
# ───────────────────────────────────────────────────────────────────────────

class BindingAffinityTask(BaseValidationTask):
    """
    Compare ECFP4-only vs ECFP4+CRAFT for multi-target pIC50 prediction.

    CRAFT adds value here because it tells the model WHICH target each
    compound-target pair belongs to (biological context).
    """

    def __init__(self, *args, include_baselines: bool = False,
                 use_extended: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_baselines = include_baselines
        self.use_extended = use_extended

    def run(self) -> Dict:
        from .target_panels import get_all_binding_targets, get_extended_binding_targets
        from .dataset_builders import BindingAffinityDatasetBuilder

        if self.use_extended:
            targets = get_extended_binding_targets()
        else:
            targets = get_all_binding_targets()
        if self.max_targets > 0:
            targets = targets[:self.max_targets]

        logger.info("=== Task 1: Binding Affinity (%d targets) ===", len(targets))
        t0 = time.time()

        # Build dataset
        builder = BindingAffinityDatasetBuilder()
        data = builder.build(targets, include_baselines=self.include_baselines)

        # Scaffold split on compounds
        train_idx, val_idx, test_idx = self._scaffold_split(data['smiles'])

        # --- Baseline: ECFP4-only ---
        logger.info("Training ECFP4-only baseline...")
        rf_base, gb_base = self._train_rf_gb_regressor(
            data['X_ecfp'][train_idx], data['y'][train_idx])
        y_pred_base_val = self._ensemble_predict_regression(
            rf_base, gb_base, data['X_ecfp'][val_idx])
        y_pred_base_test = self._ensemble_predict_regression(
            rf_base, gb_base, data['X_ecfp'][test_idx])

        # --- CRAFT-augmented: ECFP4 + CRAFT ---
        logger.info("Training ECFP4+CRAFT augmented model...")
        rf_craft, gb_craft = self._train_rf_gb_regressor(
            data['X_craft_augmented'][train_idx], data['y'][train_idx])
        y_pred_craft_val = self._ensemble_predict_regression(
            rf_craft, gb_craft, data['X_craft_augmented'][val_idx])
        y_pred_craft_test = self._ensemble_predict_regression(
            rf_craft, gb_craft, data['X_craft_augmented'][test_idx])

        # Metrics
        metrics = {
            'baseline_ecfp': {
                'val': self._regression_metrics(data['y'][val_idx], y_pred_base_val),
                'test': self._regression_metrics(data['y'][test_idx], y_pred_base_test),
            },
            'craft_augmented': {
                'val': self._regression_metrics(data['y'][val_idx], y_pred_craft_val),
                'test': self._regression_metrics(data['y'][test_idx], y_pred_craft_test),
            },
        }

        # --- Competitive baselines ---
        metrics.update(self._train_baselines(
            data, train_idx, val_idx, test_idx))

        # Delta
        delta_r2 = (metrics['craft_augmented']['test']['r2']
                     - metrics['baseline_ecfp']['test']['r2'])
        delta_mae = (metrics['craft_augmented']['test']['mae']
                      - metrics['baseline_ecfp']['test']['mae'])
        metrics['delta'] = {
            'test_r2': delta_r2,
            'test_mae': delta_mae,
            'craft_improves': delta_r2 > 0,
        }

        # Per-target breakdown
        per_target = self._per_target_breakdown(
            data, test_idx, rf_base, gb_base, rf_craft, gb_craft)
        metrics['per_target'] = per_target

        elapsed = time.time() - t0
        config = {
            'task': 'binding_affinity',
            'n_targets': len(targets),
            'n_samples': len(data['y']),
            'split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
            'rf_n_estimators': self.rf_n_estimators,
            'gb_n_estimators': self.gb_n_estimators,
            'elapsed_seconds': elapsed,
        }

        self._save_results(metrics, config, 'binding_affinity')

        logger.info("Task 1 complete: Baseline R²=%.3f, CRAFT R²=%.3f (Δ=%+.3f)",
                     metrics['baseline_ecfp']['test']['r2'],
                     metrics['craft_augmented']['test']['r2'], delta_r2)

        return metrics

    def _train_baselines(self, data, train_idx, val_idx, test_idx) -> Dict:
        """Train competitive baseline models if data is available."""
        baselines = {}
        y_train = data['y'][train_idx]
        y_test = data['y'][test_idx]

        for key, label in [('X_random_augmented', 'random_ablation'),
                           ('X_esm2_augmented', 'esm2_baseline'),
                           ('X_metadata_augmented', 'metadata_baseline')]:
            if key not in data:
                continue
            try:
                logger.info("Training %s...", label)
                rf, gb = self._train_rf_gb_regressor(
                    data[key][train_idx], y_train)
                y_pred_test = self._ensemble_predict_regression(
                    rf, gb, data[key][test_idx])
                baselines[label] = {
                    'test': self._regression_metrics(y_test, y_pred_test),
                }
            except Exception as e:
                logger.warning("Baseline %s failed: %s", label, e)

        return baselines

    def _aggregate_results(self, per_seed_results: List[Dict],
                           seeds: List[int]) -> Dict:
        """Aggregate binding affinity results across seeds."""
        from .statistical_utils import aggregate_metrics, paired_wilcoxon_test

        base_r2s = [r['baseline_ecfp']['test']['r2'] for r in per_seed_results]
        craft_r2s = [r['craft_augmented']['test']['r2'] for r in per_seed_results]
        base_maes = [r['baseline_ecfp']['test']['mae'] for r in per_seed_results]
        craft_maes = [r['craft_augmented']['test']['mae'] for r in per_seed_results]

        agg = {
            'baseline_r2': aggregate_metrics(base_r2s),
            'craft_r2': aggregate_metrics(craft_r2s),
            'baseline_mae': aggregate_metrics(base_maes),
            'craft_mae': aggregate_metrics(craft_maes),
            'wilcoxon_r2': paired_wilcoxon_test(base_r2s, craft_r2s),
            'wilcoxon_mae': paired_wilcoxon_test(craft_maes, base_maes),
        }

        # Aggregate baseline metrics if present
        for bl_key in ('random_ablation', 'esm2_baseline', 'metadata_baseline'):
            bl_r2s = [r.get(bl_key, {}).get('test', {}).get('r2')
                      for r in per_seed_results]
            bl_r2s = [v for v in bl_r2s if v is not None]
            if bl_r2s:
                agg[f'{bl_key}_r2'] = aggregate_metrics(bl_r2s)
                agg[f'wilcoxon_craft_vs_{bl_key}'] = paired_wilcoxon_test(
                    bl_r2s, craft_r2s[:len(bl_r2s)])

        return agg

    def _per_target_breakdown(self, data, test_idx, rf_base, gb_base,
                               rf_craft, gb_craft) -> Dict:
        """Compute per-target metrics on test set and save predictions."""
        target_ids = [data['target_ids'][i] for i in test_idx]
        target_genes = [data['target_genes'][i] for i in test_idx]
        y_test = data['y'][test_idx]

        per_target = {}
        predictions = {}
        unique_targets = sorted(set(target_ids))

        for tid in unique_targets:
            mask = np.array([t == tid for t in target_ids])
            if mask.sum() < 5:
                continue

            gene = target_genes[np.where(mask)[0][0]]
            y_t = y_test[mask]

            y_base = self._ensemble_predict_regression(
                rf_base, gb_base, data['X_ecfp'][test_idx][mask])
            y_craft = self._ensemble_predict_regression(
                rf_craft, gb_craft, data['X_craft_augmented'][test_idx][mask])

            label = gene or tid
            per_target[label] = {
                'n_compounds': int(mask.sum()),
                'baseline_r2': float(r2_score(y_t, y_base)) if len(y_t) > 1 else 0,
                'craft_r2': float(r2_score(y_t, y_craft)) if len(y_t) > 1 else 0,
            }
            predictions[label] = {
                'y_true': y_t,
                'y_pred_baseline': y_base,
                'y_pred_craft': y_craft,
            }

        # Save raw predictions for figure generation
        self._save_predictions(predictions, suffix=f'seed_{self.seed}')

        return per_target


# ───────────────────────────────────────────────────────────────────────────
# Task 2: ADMET Improvement (Classification)
# ───────────────────────────────────────────────────────────────────────────

class ADMETTask(BaseValidationTask):
    """
    Compare ECFP4-only vs ECFP4+CRAFT for pooled Tox21 classification.

    CRAFT encodes the biological context of each assay's target, letting
    a single model distinguish between different toxicity mechanisms.
    """

    def __init__(self, *args, include_baselines: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_baselines = include_baselines

    def run(self) -> Dict:
        from .dataset_builders import ADMETDatasetBuilder

        logger.info("=== Task 2: ADMET (Tox21) ===")
        t0 = time.time()

        builder = ADMETDatasetBuilder()
        data = builder.build(include_baselines=self.include_baselines)

        # Compound-level scaffold split
        train_mask, val_mask, test_mask = self._compound_level_scaffold_split(
            data['smiles'], data['compound_to_rows'])

        # --- Baseline: ECFP4-only ---
        logger.info("Training ECFP4-only baseline...")
        rf_base, gb_base = self._train_rf_gb_classifier(
            data['X_ecfp'][train_mask], data['y'][train_mask])
        y_proba_base_val = self._ensemble_predict_proba(
            rf_base, gb_base, data['X_ecfp'][val_mask])
        y_proba_base_test = self._ensemble_predict_proba(
            rf_base, gb_base, data['X_ecfp'][test_mask])

        # --- CRAFT-augmented ---
        logger.info("Training ECFP4+CRAFT augmented model...")
        rf_craft, gb_craft = self._train_rf_gb_classifier(
            data['X_craft_augmented'][train_mask], data['y'][train_mask])
        y_proba_craft_val = self._ensemble_predict_proba(
            rf_craft, gb_craft, data['X_craft_augmented'][val_mask])
        y_proba_craft_test = self._ensemble_predict_proba(
            rf_craft, gb_craft, data['X_craft_augmented'][test_mask])

        # Metrics
        metrics = {
            'baseline_ecfp': {
                'val': self._classification_metrics(data['y'][val_mask], y_proba_base_val),
                'test': self._classification_metrics(data['y'][test_mask], y_proba_base_test),
            },
            'craft_augmented': {
                'val': self._classification_metrics(data['y'][val_mask], y_proba_craft_val),
                'test': self._classification_metrics(data['y'][test_mask], y_proba_craft_test),
            },
        }

        # --- Competitive baselines ---
        metrics.update(self._train_baselines(
            data, train_mask, test_mask))

        delta_auc = (metrics['craft_augmented']['test'].get('auc_roc', 0)
                      - metrics['baseline_ecfp']['test'].get('auc_roc', 0))
        metrics['delta'] = {
            'test_auc_roc': delta_auc,
            'craft_improves': delta_auc > 0,
        }

        # Per-task breakdown
        per_task = self._per_task_breakdown(
            data, test_mask, rf_base, gb_base, rf_craft, gb_craft)
        metrics['per_task'] = per_task

        elapsed = time.time() - t0
        config = {
            'task': 'admet_tox21',
            'n_pairs': len(data['y']),
            'n_unique_compounds': data['metadata']['n_unique_compounds'],
            'n_tasks': data['metadata']['n_tasks'],
            'split': {
                'train': int(train_mask.sum()),
                'val': int(val_mask.sum()),
                'test': int(test_mask.sum()),
            },
            'elapsed_seconds': elapsed,
        }

        self._save_results(metrics, config, 'admet')

        logger.info("Task 2 complete: Baseline AUC=%.3f, CRAFT AUC=%.3f (Δ=%+.3f)",
                     metrics['baseline_ecfp']['test'].get('auc_roc', 0),
                     metrics['craft_augmented']['test'].get('auc_roc', 0),
                     delta_auc)

        return metrics

    def _train_baselines(self, data, train_mask, test_mask) -> Dict:
        """Train competitive baseline classifiers if data is available."""
        baselines = {}
        y_train = data['y'][train_mask]
        y_test = data['y'][test_mask]

        for key, label in [('X_random_augmented', 'random_ablation'),
                           ('X_esm2_augmented', 'esm2_baseline'),
                           ('X_metadata_augmented', 'metadata_baseline')]:
            if key not in data:
                continue
            try:
                logger.info("Training ADMET %s...", label)
                rf, gb = self._train_rf_gb_classifier(
                    data[key][train_mask], y_train)
                y_proba_test = self._ensemble_predict_proba(
                    rf, gb, data[key][test_mask])
                baselines[label] = {
                    'test': self._classification_metrics(y_test, y_proba_test),
                }
            except Exception as e:
                logger.warning("ADMET baseline %s failed: %s", label, e)

        return baselines

    def _aggregate_results(self, per_seed_results: List[Dict],
                           seeds: List[int]) -> Dict:
        """Aggregate ADMET results across seeds."""
        from .statistical_utils import aggregate_metrics, paired_wilcoxon_test

        base_aucs = [r['baseline_ecfp']['test']['auc_roc']
                     for r in per_seed_results]
        craft_aucs = [r['craft_augmented']['test']['auc_roc']
                      for r in per_seed_results]
        base_auprcs = [r['baseline_ecfp']['test'].get('auprc', float('nan'))
                       for r in per_seed_results]
        craft_auprcs = [r['craft_augmented']['test'].get('auprc', float('nan'))
                        for r in per_seed_results]

        agg = {
            'baseline_auc': aggregate_metrics(base_aucs),
            'craft_auc': aggregate_metrics(craft_aucs),
            'baseline_auprc': aggregate_metrics(base_auprcs),
            'craft_auprc': aggregate_metrics(craft_auprcs),
            'wilcoxon_auc': paired_wilcoxon_test(base_aucs, craft_aucs),
            'wilcoxon_auprc': paired_wilcoxon_test(base_auprcs, craft_auprcs),
        }

        # Aggregate baseline metrics if present
        for bl_key in ('random_ablation', 'esm2_baseline', 'metadata_baseline'):
            bl_aucs = [r.get(bl_key, {}).get('test', {}).get('auc_roc')
                       for r in per_seed_results]
            bl_aucs = [v for v in bl_aucs if v is not None]
            if bl_aucs:
                agg[f'{bl_key}_auc'] = aggregate_metrics(bl_aucs)
                agg[f'wilcoxon_craft_vs_{bl_key}'] = paired_wilcoxon_test(
                    bl_aucs, craft_aucs[:len(bl_aucs)])

        return agg

    def _per_task_breakdown(self, data, test_mask, rf_base, gb_base,
                             rf_craft, gb_craft) -> Dict:
        """Compute per-Tox21 task metrics."""
        test_tasks = [data['task_names'][i]
                      for i in range(len(data['task_names'])) if test_mask[i]]
        y_test = data['y'][test_mask]

        y_proba_base = self._ensemble_predict_proba(
            rf_base, gb_base, data['X_ecfp'][test_mask])
        y_proba_craft = self._ensemble_predict_proba(
            rf_craft, gb_craft, data['X_craft_augmented'][test_mask])

        per_task = {}
        for task_name in sorted(set(test_tasks)):
            mask = np.array([t == task_name for t in test_tasks])
            if mask.sum() < 5:
                continue
            yt = y_test[mask]
            if len(set(yt)) < 2:
                continue  # Skip if only one class

            per_task[task_name] = {
                'n_samples': int(mask.sum()),
                'baseline_auc': float(roc_auc_score(yt, y_proba_base[mask])),
                'craft_auc': float(roc_auc_score(yt, y_proba_craft[mask])),
            }

        return per_task


# ───────────────────────────────────────────────────────────────────────────
# Task 3: Selectivity Profiling (No ML)
# ───────────────────────────────────────────────────────────────────────────

class SelectivityTask(BaseValidationTask):
    """
    Show CRAFT captures target family structure:
    within-family similarity > between-family similarity.
    """

    def run(self) -> Dict:
        from .dataset_builders import SelectivityDatasetBuilder
        from scipy import stats

        logger.info("=== Task 3: Selectivity Profiling ===")
        t0 = time.time()

        builder = SelectivityDatasetBuilder()
        data = builder.build()

        within = data['within_family_similarities']
        between = data['between_family_similarities']

        # Statistical test: within > between
        if within and between:
            U_stat, p_value = stats.mannwhitneyu(
                within, between, alternative='greater')
        else:
            U_stat, p_value = 0, 1.0

        # Per-module analysis: which modules drive family clustering?
        module_analysis = {}
        for mod, mod_matrix in data['module_similarities'].items():
            mod_within = []
            mod_between = []
            n = len(data['family_labels'])
            for i in range(n):
                for j in range(i + 1, n):
                    if data['family_labels'][i] == data['family_labels'][j]:
                        mod_within.append(float(mod_matrix[i, j]))
                    else:
                        mod_between.append(float(mod_matrix[i, j]))

            if mod_within and mod_between:
                mod_U, mod_p = stats.mannwhitneyu(
                    mod_within, mod_between, alternative='greater')
            else:
                mod_U, mod_p = 0, 1.0

            module_analysis[mod] = {
                'within_mean': float(np.mean(mod_within)) if mod_within else 0,
                'between_mean': float(np.mean(mod_between)) if mod_between else 0,
                'separation': (float(np.mean(mod_within)) - float(np.mean(mod_between))
                               if mod_within and mod_between else 0),
                'U_statistic': float(mod_U),
                'p_value': float(mod_p),
            }

        metrics = {
            'overall': {
                'within_family_mean': float(np.mean(within)) if within else 0,
                'between_family_mean': float(np.mean(between)) if between else 0,
                'within_family_std': float(np.std(within)) if within else 0,
                'between_family_std': float(np.std(between)) if between else 0,
                'within_family_values': [float(v) for v in within],
                'between_family_values': [float(v) for v in between],
                'U_statistic': float(U_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
            },
            'per_module': module_analysis,
            'similarity_matrix': data['similarity_matrix'].tolist(),
            'target_labels': data['target_labels'],
            'family_labels': data['family_labels'],
            'craft_vectors': [v.tolist() for v in data['craft_vectors']],
        }

        # --- Masked analysis: zero bits 48-63 (protein family classification) ---
        # Bits 48-63 explicitly encode target family (GPCR, kinase, etc.).
        # Masking them proves family signal is distributed across other modules
        # (localization, signaling, pocket), not just hardcoded in A4 sub-block.
        logger.info("Running masked selectivity (bits 48-63 zeroed)...")
        masked_metrics = self._masked_family_analysis(
            data['craft_vectors'], data['family_labels'], stats)
        metrics['masked_no_family_bits'] = masked_metrics

        elapsed = time.time() - t0
        config = {
            'task': 'selectivity',
            'n_targets': len(data['target_labels']),
            'n_kinases': data['metadata']['n_kinases'],
            'n_gpcrs': data['metadata']['n_gpcrs'],
            'elapsed_seconds': elapsed,
        }

        self._save_results(metrics, config, 'selectivity')

        logger.info("Task 3 complete: within=%.3f, between=%.3f, p=%.2e, significant=%s",
                     metrics['overall']['within_family_mean'],
                     metrics['overall']['between_family_mean'],
                     p_value, p_value < 0.05)
        logger.info("  Masked (no family bits): within=%.3f, between=%.3f, p=%.2e",
                     masked_metrics['within_family_mean'],
                     masked_metrics['between_family_mean'],
                     masked_metrics['p_value'])

        return metrics

    def _masked_family_analysis(self, craft_vectors, family_labels, stats_module) -> Dict:
        """
        Re-run selectivity analysis with bits 48-63 (family classification) zeroed.

        Proves CRAFT captures family structure from distributed biological features
        (Modules A1-A3, B, C, D), not just the explicit family bits in A4.
        """
        # Mask the family classification bits (sub-block A4: positions 48-63)
        masked_vectors = [v.copy() for v in craft_vectors]
        for v in masked_vectors:
            v[48:64] = 0

        # Recompute Tanimoto matrix on masked vectors
        n = len(masked_vectors)
        masked_sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                a, b = masked_vectors[i], masked_vectors[j]
                intersection = np.sum(np.minimum(a, b))
                union = np.sum(np.maximum(a, b))
                sim = float(intersection / union) if union > 0 else 0.0
                masked_sim_matrix[i, j] = sim
                masked_sim_matrix[j, i] = sim

        # Collect within/between family similarities
        masked_within, masked_between = [], []
        for i in range(n):
            for j in range(i + 1, n):
                if family_labels[i] == family_labels[j]:
                    masked_within.append(float(masked_sim_matrix[i, j]))
                else:
                    masked_between.append(float(masked_sim_matrix[i, j]))

        # Mann-Whitney U test
        if masked_within and masked_between:
            U_stat, p_val = stats_module.mannwhitneyu(
                masked_within, masked_between, alternative='greater')
        else:
            U_stat, p_val = 0, 1.0

        return {
            'within_family_mean': float(np.mean(masked_within)) if masked_within else 0,
            'between_family_mean': float(np.mean(masked_between)) if masked_between else 0,
            'within_family_std': float(np.std(masked_within)) if masked_within else 0,
            'between_family_std': float(np.std(masked_between)) if masked_between else 0,
            'within_family_values': masked_within,
            'between_family_values': masked_between,
            'U_statistic': float(U_stat),
            'p_value': float(p_val),
            'significant': p_val < 0.05,
            'bits_masked': '48-63 (protein family classification, sub-block A4)',
            'n_bits_masked': 16,
        }


# ───────────────────────────────────────────────────────────────────────────
# Task 4: Polypharmacology (Classification)
# ───────────────────────────────────────────────────────────────────────────

class PolypharmacologyTask(BaseValidationTask):
    """
    Predict whether a compound is active against a target.
    CRAFT provides the biological context of the target.
    """

    def __init__(self, *args, include_baselines: bool = False,
                 use_extended: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_baselines = include_baselines
        self.use_extended = use_extended

    def run(self) -> Dict:
        from .target_panels import get_all_binding_targets, get_extended_binding_targets
        from .dataset_builders import PolypharmacologyDatasetBuilder

        if self.use_extended:
            targets = get_extended_binding_targets()
        else:
            targets = get_all_binding_targets()
        if self.max_targets > 0:
            targets = targets[:self.max_targets]

        logger.info("=== Task 4: Polypharmacology (%d targets) ===", len(targets))
        t0 = time.time()

        builder = PolypharmacologyDatasetBuilder()
        data = builder.build(targets, include_baselines=self.include_baselines)

        # Scaffold split
        train_idx, val_idx, test_idx = self._scaffold_split(data['smiles'])

        # --- Baseline: ECFP4-only ---
        logger.info("Training ECFP4-only baseline...")
        rf_base, gb_base = self._train_rf_gb_classifier(
            data['X_ecfp'][train_idx], data['y'][train_idx])
        y_proba_base_test = self._ensemble_predict_proba(
            rf_base, gb_base, data['X_ecfp'][test_idx])

        # --- One-hot target baseline ---
        # Encodes target identity (arbitrary index) without biological meaning.
        # If CRAFT outperforms one-hot, it proves biological encoding adds value
        # beyond simple target disambiguation.
        logger.info("Training ECFP4+one-hot target baseline...")
        unique_targets = sorted(set(data['target_ids']))
        target_to_idx = {t: i for i, t in enumerate(unique_targets)}
        n_targets = len(unique_targets)
        onehot_matrix = np.zeros((len(data['y']), n_targets), dtype=np.float32)
        for i, tid in enumerate(data['target_ids']):
            onehot_matrix[i, target_to_idx[tid]] = 1.0
        X_onehot = np.hstack([data['X_ecfp'], onehot_matrix])

        rf_onehot, gb_onehot = self._train_rf_gb_classifier(
            X_onehot[train_idx], data['y'][train_idx])
        y_proba_onehot_test = self._ensemble_predict_proba(
            rf_onehot, gb_onehot, X_onehot[test_idx])

        # --- CRAFT-augmented ---
        logger.info("Training ECFP4+CRAFT augmented model...")
        rf_craft, gb_craft = self._train_rf_gb_classifier(
            data['X_craft_augmented'][train_idx], data['y'][train_idx])
        y_proba_craft_test = self._ensemble_predict_proba(
            rf_craft, gb_craft, data['X_craft_augmented'][test_idx])

        metrics = {
            'baseline_ecfp': {
                'test': self._classification_metrics(
                    data['y'][test_idx], y_proba_base_test),
            },
            'onehot_baseline': {
                'test': self._classification_metrics(
                    data['y'][test_idx], y_proba_onehot_test),
                'n_onehot_features': n_targets,
            },
            'craft_augmented': {
                'test': self._classification_metrics(
                    data['y'][test_idx], y_proba_craft_test),
            },
        }

        # --- Competitive baselines ---
        metrics.update(self._train_baselines(
            data, train_idx, test_idx))

        # Deltas
        base_auc = metrics['baseline_ecfp']['test'].get('auc_roc', 0)
        onehot_auc = metrics['onehot_baseline']['test'].get('auc_roc', 0)
        craft_auc = metrics['craft_augmented']['test'].get('auc_roc', 0)

        metrics['delta'] = {
            'craft_vs_baseline_auc': craft_auc - base_auc,
            'onehot_vs_baseline_auc': onehot_auc - base_auc,
            'craft_vs_onehot_auc': craft_auc - onehot_auc,
            'test_auprc': (metrics['craft_augmented']['test'].get('auprc', 0)
                            - metrics['baseline_ecfp']['test'].get('auprc', 0)),
            'craft_improves_over_baseline': craft_auc > base_auc,
            'craft_improves_over_onehot': craft_auc > onehot_auc,
        }

        elapsed = time.time() - t0
        config = {
            'task': 'polypharmacology',
            'n_targets': len(targets),
            'n_samples': len(data['y']),
            'n_positive': data['metadata']['n_positive'],
            'n_negative': data['metadata']['n_negative'],
            'split': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
            'elapsed_seconds': elapsed,
        }

        self._save_results(metrics, config, 'polypharmacology')

        logger.info("Task 4 complete: Baseline AUC=%.3f, One-hot AUC=%.3f, CRAFT AUC=%.3f",
                     base_auc, onehot_auc, craft_auc)
        logger.info("  CRAFT vs baseline: %+.3f | CRAFT vs one-hot: %+.3f",
                     craft_auc - base_auc, craft_auc - onehot_auc)

        return metrics

    def _train_baselines(self, data, train_idx, test_idx) -> Dict:
        """Train competitive baseline classifiers if data is available."""
        baselines = {}
        y_train = data['y'][train_idx]
        y_test = data['y'][test_idx]

        for key, label in [('X_random_augmented', 'random_ablation'),
                           ('X_esm2_augmented', 'esm2_baseline'),
                           ('X_metadata_augmented', 'metadata_baseline')]:
            if key not in data:
                continue
            try:
                logger.info("Training polypharm %s...", label)
                rf, gb = self._train_rf_gb_classifier(
                    data[key][train_idx], y_train)
                y_proba_test = self._ensemble_predict_proba(
                    rf, gb, data[key][test_idx])
                baselines[label] = {
                    'test': self._classification_metrics(y_test, y_proba_test),
                }
            except Exception as e:
                logger.warning("Polypharm baseline %s failed: %s", label, e)

        return baselines

    def _aggregate_results(self, per_seed_results: List[Dict],
                           seeds: List[int]) -> Dict:
        """Aggregate polypharmacology results across seeds."""
        from .statistical_utils import aggregate_metrics, paired_wilcoxon_test

        base_aucs = [r['baseline_ecfp']['test']['auc_roc']
                     for r in per_seed_results]
        onehot_aucs = [r['onehot_baseline']['test']['auc_roc']
                       for r in per_seed_results]
        craft_aucs = [r['craft_augmented']['test']['auc_roc']
                      for r in per_seed_results]

        agg = {
            'baseline_auc': aggregate_metrics(base_aucs),
            'onehot_auc': aggregate_metrics(onehot_aucs),
            'craft_auc': aggregate_metrics(craft_aucs),
            'wilcoxon_craft_vs_baseline': paired_wilcoxon_test(
                base_aucs, craft_aucs),
            'wilcoxon_craft_vs_onehot': paired_wilcoxon_test(
                onehot_aucs, craft_aucs),
        }

        # Aggregate competitive baseline metrics if present
        for bl_key in ('random_ablation', 'esm2_baseline', 'metadata_baseline'):
            bl_aucs = [r.get(bl_key, {}).get('test', {}).get('auc_roc')
                       for r in per_seed_results]
            bl_aucs = [v for v in bl_aucs if v is not None]
            if bl_aucs:
                agg[f'{bl_key}_auc'] = aggregate_metrics(bl_aucs)
                agg[f'wilcoxon_craft_vs_{bl_key}'] = paired_wilcoxon_test(
                    bl_aucs, craft_aucs[:len(bl_aucs)])

        return agg


# ───────────────────────────────────────────────────────────────────────────
# Task 5: Standard-Format Tox21 (Per-Task Benchmarking)
# ───────────────────────────────────────────────────────────────────────────

class StandardTox21Task(BaseValidationTask):
    """
    Standard per-task Tox21 classification - 12 independent classifiers.

    CRAFT is NOT expected to help here because in per-task mode every compound
    within a task gets the same CRAFT vector (constant feature → no information
    gain). This task serves two purposes:

    1. Establish baseline comparability with published Tox21 benchmarks
       (e.g., MoleculeNet, Wu et al. 2018)
    2. Confirm that CRAFT's value is specific to multi-target settings
       (comparison with Task 2 pooled format)
    """

    def run(self) -> Dict:
        import deepchem as dc

        logger.info("=== Task 5: Standard Tox21 (Per-Task) ===")
        t0 = time.time()

        # Load Tox21 via DeepChem
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer='ECFP', splitter='scaffold', reload=False)
        train_ds, valid_ds, test_ds = datasets

        # Pool all splits - we re-split with our scaffold splitter
        all_smiles = (list(train_ds.ids) + list(valid_ds.ids)
                      + list(test_ds.ids))
        all_y = np.vstack([train_ds.y, valid_ds.y, test_ds.y])
        # DeepChem marks unmeasured results with y = 0, w = 0 (never NaN), so
        # the weight, not NaN, identifies a measured label. Same fix as
        # ADMETDatasetBuilder.
        all_w = np.vstack([train_ds.w, valid_ds.w, test_ds.w])

        logger.info("Tox21 loaded: %d compounds, %d tasks, %d measured labels "
                    "(%d unmeasured excluded)",
                    len(all_smiles), len(tasks),
                    int((all_w > 0).sum()), int((all_w == 0).sum()))

        # Generate ECFP4 once for all compounds
        from .featurizers import CRAFTTrainingFeaturizer
        featurizer = CRAFTTrainingFeaturizer()
        X_ecfp = featurizer.featurize_ecfp(all_smiles)

        # Scaffold split at compound level (same split for all tasks)
        train_idx, val_idx, test_idx = self._scaffold_split(all_smiles)

        per_task_results = {}
        all_aucs = []

        for task_idx, task_name in enumerate(tasks):
            # Extract MEASURED labels for this task (w > 0)
            labels = all_y[:, task_idx]
            valid_mask = (all_w[:, task_idx] > 0) & ~np.isnan(labels)

            # Intersect with split indices
            train_valid = [i for i in train_idx if valid_mask[i]]
            test_valid = [i for i in test_idx if valid_mask[i]]

            if len(train_valid) < 50 or len(test_valid) < 10:
                logger.warning("Skipping %s: too few samples (train=%d, test=%d)",
                               task_name, len(train_valid), len(test_valid))
                continue

            y_train = labels[train_valid].astype(np.float32)
            y_test = labels[test_valid].astype(np.float32)

            # Check class balance
            if len(set(y_train)) < 2 or len(set(y_test)) < 2:
                logger.warning("Skipping %s: single class in train or test",
                               task_name)
                continue

            n_pos_train = int(y_train.sum())
            n_pos_test = int(y_test.sum())

            # Train ECFP4-only classifier for this task
            try:
                rf, gb = self._train_rf_gb_classifier(
                    X_ecfp[train_valid], y_train)
                y_proba_test = self._ensemble_predict_proba(
                    rf, gb, X_ecfp[test_valid])
                task_metrics = self._classification_metrics(y_test, y_proba_test)
            except Exception as e:
                logger.warning("Training failed for %s: %s", task_name, e)
                continue

            per_task_results[task_name] = {
                'n_train': len(train_valid),
                'n_test': len(test_valid),
                'n_pos_train': n_pos_train,
                'n_pos_test': n_pos_test,
                'pos_rate_train': n_pos_train / len(train_valid),
                'pos_rate_test': n_pos_test / len(test_valid),
                **task_metrics,
            }

            auc = task_metrics.get('auc_roc', float('nan'))
            if not np.isnan(auc):
                all_aucs.append(auc)

            logger.info("  %s: AUC=%.3f (n_train=%d, n_test=%d, pos=%.1f%%)",
                        task_name, auc, len(train_valid), len(test_valid),
                        100 * n_pos_test / len(test_valid))

        mean_auc = float(np.mean(all_aucs)) if all_aucs else 0
        std_auc = float(np.std(all_aucs)) if all_aucs else 0

        metrics = {
            'per_task': per_task_results,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'n_tasks_evaluated': len(per_task_results),
        }

        elapsed = time.time() - t0
        config = {
            'task': 'standard_tox21',
            'n_compounds': len(all_smiles),
            'n_tasks': len(tasks),
            'n_tasks_evaluated': len(per_task_results),
            'split': {
                'train': len(train_idx),
                'val': len(val_idx),
                'test': len(test_idx),
            },
            'elapsed_seconds': elapsed,
        }

        self._save_results(metrics, config, 'standard_tox21')

        logger.info("Task 5 complete: Mean AUC=%.3f (+/- %.3f) across %d tasks",
                     mean_auc, std_auc, len(per_task_results))

        return metrics

    def _aggregate_results(self, per_seed_results: List[Dict],
                           seeds: List[int]) -> Dict:
        """Aggregate standard Tox21 results across seeds."""
        from .statistical_utils import aggregate_metrics

        mean_aucs = [r.get('mean_auc', float('nan'))
                     for r in per_seed_results]
        mean_aucs = [v for v in mean_aucs if not np.isnan(v)]

        agg = {
            'mean_auc': aggregate_metrics(mean_aucs) if mean_aucs else {},
        }

        # Per-task aggregation across seeds
        all_task_names = set()
        for r in per_seed_results:
            all_task_names.update(r.get('per_task', {}).keys())

        per_task_agg = {}
        for tn in sorted(all_task_names):
            task_aucs = [r.get('per_task', {}).get(tn, {}).get('auc_roc')
                         for r in per_seed_results]
            task_aucs = [v for v in task_aucs
                         if v is not None and not np.isnan(v)]
            if task_aucs:
                per_task_agg[tn] = aggregate_metrics(task_aucs)

        agg['per_task'] = per_task_agg
        return agg


# ───────────────────────────────────────────────────────────────────────────
# Task 6: Leave-One-Family-Out (LOFO) Generalization
# ───────────────────────────────────────────────────────────────────────────

class LOFOTask(BaseValidationTask):
    """
    Leave-One-Family-Out: train on N-1 families, test on held-out family.

    Key hypothesis: One-hot FAILS for unseen family (all-zeros for unknown
    targets → effectively ECFP-only). CRAFT generalizes because biological
    features (localization, signaling, pocket) are shared across families.

    Tests 5 conditions per held-out family:
    1. ECFP4-only (no target info)
    2. ECFP4 + one-hot (zeros for held-out → same as ECFP-only)
    3. ECFP4 + CRAFT (biological features transfer)
    4. ECFP4 + random-256-bit (dimensionality ablation)
    5. ECFP4 + ESM-2 (sequence features transfer) [if available]

    Requires extended target panel (Phase 11) with ≥3 targets per family.
    """

    MIN_FAMILY_SIZE = 3

    def run(self) -> Dict:
        from .target_panels import get_extended_binding_targets, get_all_families
        from .dataset_builders import LOFODatasetBuilder

        logger.info("=== Task 6: LOFO Generalization ===")
        t0 = time.time()

        all_targets = get_extended_binding_targets()
        if self.max_targets > 0:
            all_targets = all_targets[:self.max_targets]

        families = get_all_families()
        builder = LOFODatasetBuilder()

        # Fold-level checkpoint directory
        fold_ckpt_dir = os.path.join(self.output_dir, 'fold_checkpoints')
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        lofo_results = {}
        self._log_memory('LOFO start')

        for held_out_family in families:
            # Memory guard before each fold
            if not self._check_memory(f'before fold {held_out_family}'):
                logger.error(
                    "ABORTING LOFO: memory too high before fold %s. "
                    "Completed folds are checkpointed - resume after restart.",
                    held_out_family)
                break

            # Partition targets
            test_targets = [t for t in all_targets
                            if t['family'] == held_out_family]
            train_targets = [t for t in all_targets
                             if t['family'] != held_out_family]

            if len(test_targets) < self.MIN_FAMILY_SIZE:
                logger.info("Skipping %s (only %d targets, need %d)",
                            held_out_family, len(test_targets),
                            self.MIN_FAMILY_SIZE)
                continue

            # Resume: check fold checkpoint
            fold_ckpt_file = os.path.join(
                fold_ckpt_dir,
                f'seed_{self.seed}_{held_out_family}.json')
            if os.path.exists(fold_ckpt_file):
                logger.info("Resuming fold %s (seed %d) from checkpoint",
                            held_out_family, self.seed)
                with open(fold_ckpt_file) as f:
                    lofo_results[held_out_family] = json.load(f)
                continue

            logger.info("LOFO: held-out=%s (%d targets), train=%d targets",
                        held_out_family, len(test_targets), len(train_targets))

            try:
                # Build datasets
                train_data = builder.build_train(train_targets)
                test_data = builder.build_test(test_targets)
            except Exception as e:
                logger.warning("LOFO dataset build failed for %s: %s",
                               held_out_family, e)
                continue

            self._log_memory(f'after dataset build ({held_out_family})')

            if len(train_data['y']) < 50 or len(test_data['y']) < 20:
                logger.warning("Skipping %s: insufficient data (train=%d, test=%d)",
                               held_out_family, len(train_data['y']),
                               len(test_data['y']))
                continue

            # Memory check after dataset build (most memory-intensive part)
            if not self._check_memory(f'after building {held_out_family} data'):
                logger.error(
                    "ABORTING fold %s: memory too high after dataset build. "
                    "Checkpointed folds preserved.", held_out_family)
                del train_data, test_data
                gc.collect()
                break

            # Build one-hot matrices
            train_target_ids = train_data['target_ids']
            test_target_ids = test_data['target_ids']
            all_train_target_set = sorted(set(train_target_ids))

            onehot_train = builder.build_onehot_matrix(
                train_target_ids, all_train_target_set)
            onehot_test = builder.build_onehot_matrix(
                test_target_ids, all_train_target_set)  # All zeros!

            X_onehot_train = np.hstack([train_data['X_ecfp'], onehot_train])
            X_onehot_test = np.hstack([test_data['X_ecfp'], onehot_test])

            # Scaffold split WITHIN training data
            train_idx, val_idx, _ = self._scaffold_split(train_data['smiles'])

            # Subsample training if too large (>25K) - keeps LOFO tractable
            MAX_LOFO_TRAIN = 25000
            if len(train_idx) > MAX_LOFO_TRAIN:
                rng = np.random.RandomState(self.seed)
                train_idx = rng.choice(
                    train_idx, size=MAX_LOFO_TRAIN, replace=False)
                train_idx.sort()
                logger.info("Subsampled training to %d (from %d)",
                            MAX_LOFO_TRAIN, len(train_idx))

            y_train = train_data['y'][train_idx]
            y_test = test_data['y']  # All test data from held-out family

            family_results = {
                'n_train_targets': len(train_targets),
                'n_test_targets': len(test_targets),
                'n_train_samples': len(y_train),
                'n_test_samples': len(y_test),
            }

            # Condition 1: ECFP4-only
            try:
                rf, gb = self._train_rf_gb_regressor(
                    train_data['X_ecfp'][train_idx], y_train)
                y_pred = self._ensemble_predict_regression(
                    rf, gb, test_data['X_ecfp'])
                family_results['ecfp_only'] = self._regression_metrics(
                    y_test, y_pred)
                del rf, gb, y_pred; gc.collect()
                logger.info("  [1/5] ECFP-only done: R2=%.3f",
                            family_results['ecfp_only']['r2'])
            except Exception as e:
                logger.warning("ECFP-only failed for %s: %s", held_out_family, e)
                family_results['ecfp_only'] = {'r2': float('nan')}

            # Condition 2: ECFP4 + one-hot (zeros for held-out)
            try:
                rf, gb = self._train_rf_gb_regressor(
                    X_onehot_train[train_idx], y_train)
                y_pred = self._ensemble_predict_regression(
                    rf, gb, X_onehot_test)
                family_results['onehot'] = self._regression_metrics(
                    y_test, y_pred)
                del rf, gb, y_pred; gc.collect()
                logger.info("  [2/5] One-hot done: R2=%.3f",
                            family_results['onehot']['r2'])
            except Exception as e:
                logger.warning("One-hot failed for %s: %s", held_out_family, e)
                family_results['onehot'] = {'r2': float('nan')}

            # Condition 3: ECFP4 + CRAFT
            try:
                rf, gb = self._train_rf_gb_regressor(
                    train_data['X_craft_augmented'][train_idx], y_train)
                y_pred = self._ensemble_predict_regression(
                    rf, gb, test_data['X_craft_augmented'])
                family_results['craft'] = self._regression_metrics(
                    y_test, y_pred)
                del rf, gb, y_pred; gc.collect()
                logger.info("  [3/5] CRAFT done: R2=%.3f",
                            family_results['craft']['r2'])
            except Exception as e:
                logger.warning("CRAFT failed for %s: %s", held_out_family, e)
                family_results['craft'] = {'r2': float('nan')}

            # Condition 4: ECFP4 + random-256-bit
            if 'X_random_augmented' in train_data and 'X_random_augmented' in test_data:
                try:
                    rf, gb = self._train_rf_gb_regressor(
                        train_data['X_random_augmented'][train_idx], y_train)
                    y_pred = self._ensemble_predict_regression(
                        rf, gb, test_data['X_random_augmented'])
                    family_results['random'] = self._regression_metrics(
                        y_test, y_pred)
                    del rf, gb, y_pred; gc.collect()
                    logger.info("  [4/5] Random done: R2=%.3f",
                                family_results['random']['r2'])
                except Exception as e:
                    logger.warning("Random failed for %s: %s", held_out_family, e)

            # Condition 5: ECFP4 + ESM-2
            if 'X_esm2_augmented' in train_data and 'X_esm2_augmented' in test_data:
                try:
                    rf, gb = self._train_rf_gb_regressor(
                        train_data['X_esm2_augmented'][train_idx], y_train)
                    y_pred = self._ensemble_predict_regression(
                        rf, gb, test_data['X_esm2_augmented'])
                    family_results['esm2'] = self._regression_metrics(
                        y_test, y_pred)
                    del rf, gb, y_pred; gc.collect()
                    logger.info("  [5/5] ESM-2 done: R2=%.3f",
                                family_results['esm2']['r2'])
                except Exception as e:
                    logger.warning("ESM-2 failed for %s: %s", held_out_family, e)

            lofo_results[held_out_family] = family_results

            # Save fold checkpoint
            with open(fold_ckpt_file, 'w') as f:
                json.dump(family_results, f, indent=2, default=str)
            logger.info("Fold checkpoint saved: seed %d, %s",
                        self.seed, held_out_family)

            # Log summary for this family
            ecfp_r2 = family_results.get('ecfp_only', {}).get('r2', float('nan'))
            onehot_r2 = family_results.get('onehot', {}).get('r2', float('nan'))
            craft_r2 = family_results.get('craft', {}).get('r2', float('nan'))
            logger.info("  %s: ECFP=%.3f, OneHot=%.3f, CRAFT=%.3f",
                        held_out_family, ecfp_r2, onehot_r2, craft_r2)

            # Memory cleanup between folds
            del train_data, test_data, onehot_train, onehot_test
            del X_onehot_train, X_onehot_test
            gc.collect()
            self._log_memory(f'after fold cleanup ({held_out_family})')

        # Summary
        summary = self._compute_lofo_summary(lofo_results)

        metrics = {
            'per_family': lofo_results,
            'summary': summary,
            'n_families_tested': len(lofo_results),
        }

        elapsed = time.time() - t0
        config = {
            'task': 'lofo',
            'n_families_tested': len(lofo_results),
            'families_tested': list(lofo_results.keys()),
            'min_family_size': self.MIN_FAMILY_SIZE,
            'elapsed_seconds': elapsed,
        }

        self._save_results(metrics, config, 'lofo')

        logger.info("Task 6 complete: %d families tested", len(lofo_results))
        return metrics

    def _compute_lofo_summary(self, lofo_results: Dict) -> Dict:
        """Compute summary statistics across all LOFO folds."""
        conditions = ['ecfp_only', 'onehot', 'craft', 'random', 'esm2']
        summary = {}

        for cond in conditions:
            r2s = []
            for fam, res in lofo_results.items():
                r2 = res.get(cond, {}).get('r2')
                if r2 is not None and not np.isnan(r2):
                    r2s.append(r2)
            if r2s:
                summary[cond] = {
                    'mean_r2': float(np.mean(r2s)),
                    'std_r2': float(np.std(r2s)),
                    'n_folds': len(r2s),
                }

        # Key comparisons
        craft_r2s = [lofo_results[f].get('craft', {}).get('r2', float('nan'))
                     for f in lofo_results]
        onehot_r2s = [lofo_results[f].get('onehot', {}).get('r2', float('nan'))
                      for f in lofo_results]
        ecfp_r2s = [lofo_results[f].get('ecfp_only', {}).get('r2', float('nan'))
                    for f in lofo_results]

        # Filter NaN
        valid_pairs = [(c, o, e) for c, o, e
                       in zip(craft_r2s, onehot_r2s, ecfp_r2s)
                       if not (np.isnan(c) or np.isnan(o) or np.isnan(e))]

        if valid_pairs:
            craft_vals = [p[0] for p in valid_pairs]
            onehot_vals = [p[1] for p in valid_pairs]
            ecfp_vals = [p[2] for p in valid_pairs]

            summary['craft_vs_onehot_delta'] = float(
                np.mean(craft_vals) - np.mean(onehot_vals))
            summary['craft_vs_ecfp_delta'] = float(
                np.mean(craft_vals) - np.mean(ecfp_vals))
            summary['onehot_vs_ecfp_delta'] = float(
                np.mean(onehot_vals) - np.mean(ecfp_vals))

            # Key insight: onehot ≈ ecfp (zeros for unseen family)
            summary['onehot_degrades_to_ecfp'] = abs(
                np.mean(onehot_vals) - np.mean(ecfp_vals)) < 0.02

        return summary

    def _aggregate_results(self, per_seed_results: List[Dict],
                           seeds: List[int]) -> Dict:
        """Aggregate LOFO results across seeds."""
        from .statistical_utils import aggregate_metrics

        # Aggregate per-family, per-condition R² across seeds
        all_families = set()
        for r in per_seed_results:
            all_families.update(r.get('per_family', {}).keys())

        per_family_agg = {}
        for fam in sorted(all_families):
            fam_agg = {}
            for cond in ['ecfp_only', 'onehot', 'craft', 'random', 'esm2']:
                r2s = [r.get('per_family', {}).get(fam, {}).get(cond, {}).get('r2')
                       for r in per_seed_results]
                r2s = [v for v in r2s if v is not None and not np.isnan(v)]
                if r2s:
                    fam_agg[f'{cond}_r2'] = aggregate_metrics(r2s)
            per_family_agg[fam] = fam_agg

        return {'per_family': per_family_agg}
