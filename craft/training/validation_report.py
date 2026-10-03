"""
CRAFT Validation Report - Aggregate results from all tasks.

Generates validation_summary.json and prints formatted console output.
Supports both single-seed and multi-seed result formats with CI display.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)


def _fmt_ci(agg: Dict, fmt: str = '.3f') -> str:
    """Format aggregated metrics as 'mean [ci_low, ci_high]'."""
    if not agg or 'mean' not in agg:
        return 'N/A'
    from .statistical_utils import format_metric_with_ci
    return format_metric_with_ci(agg['mean'], agg['ci95_low'], agg['ci95_high'], fmt)


def _is_multi_seed(result: Dict) -> bool:
    """Check if a result dict is from multi-seed run."""
    return result.get('multi_seed', False)


class ValidationReport:
    """Generate CRAFT validation summary from task results."""

    def generate(self, results: Dict[str, Dict], output_path: str):
        """
        Compile all task results into a summary JSON.

        Args:
            results: Dict mapping task keys to their metrics dicts
            output_path: Path for validation_summary.json
        """
        summary = {
            'craft_version': '1.0',
            'validation_date': datetime.now().isoformat(),
            'tasks_completed': list(results.keys()),
            'summary': {},
            'per_task_details': results,
        }

        # Extract key metrics for summary - handle both single and multi-seed
        if 'task1' in results:
            t1 = results['task1']
            if _is_multi_seed(t1):
                agg = t1.get('aggregated', {})
                summary['summary']['task1_binding_affinity'] = {
                    'baseline_r2': agg.get('baseline_r2', {}),
                    'craft_r2': agg.get('craft_r2', {}),
                    'wilcoxon': agg.get('wilcoxon_r2', {}),
                    'multi_seed': True,
                }
            else:
                summary['summary']['task1_binding_affinity'] = {
                    'baseline_r2': t1.get('baseline_ecfp', {}).get('test', {}).get('r2'),
                    'craft_r2': t1.get('craft_augmented', {}).get('test', {}).get('r2'),
                    'delta_r2': t1.get('delta', {}).get('test_r2'),
                    'improves': t1.get('delta', {}).get('craft_improves', False),
                }

        if 'task2' in results:
            t2 = results['task2']
            if _is_multi_seed(t2):
                agg = t2.get('aggregated', {})
                summary['summary']['task2_admet'] = {
                    'baseline_auc': agg.get('baseline_auc', {}),
                    'craft_auc': agg.get('craft_auc', {}),
                    'wilcoxon': agg.get('wilcoxon_auc', {}),
                    'multi_seed': True,
                }
            else:
                summary['summary']['task2_admet'] = {
                    'baseline_auc': t2.get('baseline_ecfp', {}).get('test', {}).get('auc_roc'),
                    'craft_auc': t2.get('craft_augmented', {}).get('test', {}).get('auc_roc'),
                    'delta_auc': t2.get('delta', {}).get('test_auc_roc'),
                    'improves': t2.get('delta', {}).get('craft_improves', False),
                }

        if 'task3' in results:
            t3 = results['task3']
            # Task 3 (selectivity) is deterministic - no multi-seed needed
            if _is_multi_seed(t3):
                # Use first seed's result for selectivity (deterministic)
                first_seed = list(t3.get('per_seed', {}).values())[0] if t3.get('per_seed') else t3
                summary['summary']['task3_selectivity'] = {
                    'within_family_mean': first_seed.get('overall', {}).get('within_family_mean'),
                    'between_family_mean': first_seed.get('overall', {}).get('between_family_mean'),
                    'p_value': first_seed.get('overall', {}).get('p_value'),
                    'significant': first_seed.get('overall', {}).get('significant', False),
                }
            else:
                summary['summary']['task3_selectivity'] = {
                    'within_family_mean': t3.get('overall', {}).get('within_family_mean'),
                    'between_family_mean': t3.get('overall', {}).get('between_family_mean'),
                    'p_value': t3.get('overall', {}).get('p_value'),
                    'significant': t3.get('overall', {}).get('significant', False),
                }

        if 'task4' in results:
            t4 = results['task4']
            if _is_multi_seed(t4):
                agg = t4.get('aggregated', {})
                summary['summary']['task4_polypharmacology'] = {
                    'baseline_auc': agg.get('baseline_auc', {}),
                    'onehot_auc': agg.get('onehot_auc', {}),
                    'craft_auc': agg.get('craft_auc', {}),
                    'wilcoxon_vs_baseline': agg.get('wilcoxon_craft_vs_baseline', {}),
                    'wilcoxon_vs_onehot': agg.get('wilcoxon_craft_vs_onehot', {}),
                    'multi_seed': True,
                }
            else:
                d4 = t4.get('delta', {})
                craft_vs_base = d4.get('craft_vs_baseline_auc', d4.get('test_auc_roc'))
                summary['summary']['task4_polypharmacology'] = {
                    'baseline_auc': t4.get('baseline_ecfp', {}).get('test', {}).get('auc_roc'),
                    'onehot_auc': t4.get('onehot_baseline', {}).get('test', {}).get('auc_roc'),
                    'craft_auc': t4.get('craft_augmented', {}).get('test', {}).get('auc_roc'),
                    'craft_vs_baseline_auc': craft_vs_base,
                    'craft_vs_onehot_auc': d4.get('craft_vs_onehot_auc'),
                    'onehot_vs_baseline_auc': d4.get('onehot_vs_baseline_auc'),
                    'improves_over_baseline': d4.get('craft_improves_over_baseline',
                                                      d4.get('craft_improves', False)),
                    'improves_over_onehot': d4.get('craft_improves_over_onehot', False),
                }

        # Conclusion
        n_improves = 0
        n_tasks = len(summary['summary'])
        for v in summary['summary'].values():
            if not isinstance(v, dict):
                continue
            if v.get('multi_seed'):
                # Check Wilcoxon significance
                for key in ('wilcoxon', 'wilcoxon_vs_baseline'):
                    w = v.get(key, {})
                    if w.get('significant'):
                        n_improves += 1
                        break
            elif v.get('improves') or v.get('improves_over_baseline'):
                n_improves += 1
            elif v.get('significant'):
                n_improves += 1

        summary['conclusion'] = (
            f"CRAFT improves predictions in {n_improves}/{n_tasks} validation tasks."
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info("Validation summary saved to %s", output_path)

    def print_summary(self, results: Dict[str, Dict]):
        """Print formatted summary to console."""
        print("\n" + "=" * 70)
        print("  CRAFT VALIDATION FRAMEWORK - RESULTS SUMMARY")
        print("=" * 70)

        if 'task1' in results:
            t1 = results['task1']
            print(f"\n  Task 1: Binding Affinity (Regression)")
            if _is_multi_seed(t1):
                agg = t1.get('aggregated', {})
                print(f"    Baseline (ECFP4):      R² = {_fmt_ci(agg.get('baseline_r2'))}")
                print(f"    CRAFT-augmented:       R² = {_fmt_ci(agg.get('craft_r2'))}")
                w = agg.get('wilcoxon_r2', {})
                print(f"    Delta (mean):          {w.get('mean_delta', 0):+.3f}"
                      f"  (Wilcoxon p={w.get('p_value', 1):.3f})")
            else:
                d = t1.get('delta', {})
                print(f"    Baseline (ECFP4):      R² = "
                      f"{t1.get('baseline_ecfp', {}).get('test', {}).get('r2', 'N/A'):.3f}")
                print(f"    CRAFT-augmented:       R² = "
                      f"{t1.get('craft_augmented', {}).get('test', {}).get('r2', 'N/A'):.3f}")
                print(f"    Delta:                 {d.get('test_r2', 0):+.3f}"
                      f"  {'[+]' if d.get('craft_improves') else '[-]'}")

        if 'task2' in results:
            t2 = results['task2']
            print(f"\n  Task 2: ADMET (Tox21 Classification)")
            if _is_multi_seed(t2):
                agg = t2.get('aggregated', {})
                print(f"    Baseline (ECFP4):      AUC = {_fmt_ci(agg.get('baseline_auc'))}")
                print(f"    CRAFT-augmented:       AUC = {_fmt_ci(agg.get('craft_auc'))}")
                w = agg.get('wilcoxon_auc', {})
                print(f"    Delta (mean):          {w.get('mean_delta', 0):+.3f}"
                      f"  (Wilcoxon p={w.get('p_value', 1):.3f})")
            else:
                d = t2.get('delta', {})
                print(f"    Baseline (ECFP4):      AUC = "
                      f"{t2.get('baseline_ecfp', {}).get('test', {}).get('auc_roc', 'N/A'):.3f}")
                print(f"    CRAFT-augmented:       AUC = "
                      f"{t2.get('craft_augmented', {}).get('test', {}).get('auc_roc', 'N/A'):.3f}")
                print(f"    Delta:                 {d.get('test_auc_roc', 0):+.3f}"
                      f"  {'[+]' if d.get('craft_improves') else '[-]'}")

        if 'task3' in results:
            t3 = results['task3']
            # Task 3 is deterministic; if multi-seed, use first seed
            if _is_multi_seed(t3):
                t3 = list(t3.get('per_seed', {}).values())[0] if t3.get('per_seed') else t3
            o = t3.get('overall', {})
            print(f"\n  Task 3: Selectivity Profiling")
            print(f"    Within-family sim:     {o.get('within_family_mean', 0):.3f} "
                  f"(+/- {o.get('within_family_std', 0):.3f})")
            print(f"    Between-family sim:    {o.get('between_family_mean', 0):.3f} "
                  f"(+/- {o.get('between_family_std', 0):.3f})")
            print(f"    Mann-Whitney p-value:  {o.get('p_value', 1):.2e}"
                  f"  {'[significant]' if o.get('significant') else '[not significant]'}")

            per_mod = t3.get('per_module', {})
            if per_mod:
                print(f"    Per-module separation:")
                for mod in sorted(per_mod.keys()):
                    sep = per_mod[mod]['separation']
                    p = per_mod[mod]['p_value']
                    print(f"      Module {mod}: separation={sep:+.3f}, p={p:.2e}")

            masked = t3.get('masked_no_family_bits', {})
            if masked:
                print(f"    Masked (no family bits {masked.get('bits_masked', '48-63')}):")
                print(f"      Within-family:  {masked.get('within_family_mean', 0):.3f}")
                print(f"      Between-family: {masked.get('between_family_mean', 0):.3f}")
                print(f"      p-value:        {masked.get('p_value', 1):.2e}"
                      f"  {'[significant]' if masked.get('significant') else '[not significant]'}")

        if 'task4' in results:
            t4 = results['task4']
            print(f"\n  Task 4: Polypharmacology (Classification)")
            if _is_multi_seed(t4):
                agg = t4.get('aggregated', {})
                print(f"    Baseline (ECFP4):      AUC = {_fmt_ci(agg.get('baseline_auc'))}")
                print(f"    One-hot target:        AUC = {_fmt_ci(agg.get('onehot_auc'))}")
                print(f"    CRAFT-augmented:       AUC = {_fmt_ci(agg.get('craft_auc'))}")
                w_base = agg.get('wilcoxon_craft_vs_baseline', {})
                w_onehot = agg.get('wilcoxon_craft_vs_onehot', {})
                print(f"    CRAFT vs baseline:     {w_base.get('mean_delta', 0):+.3f}"
                      f"  (Wilcoxon p={w_base.get('p_value', 1):.3f})")
                print(f"    CRAFT vs one-hot:      {w_onehot.get('mean_delta', 0):+.3f}"
                      f"  (Wilcoxon p={w_onehot.get('p_value', 1):.3f})")
            else:
                d = t4.get('delta', {})
                print(f"    Baseline (ECFP4):      AUC = "
                      f"{t4.get('baseline_ecfp', {}).get('test', {}).get('auc_roc', 'N/A'):.3f}")
                onehot = t4.get('onehot_baseline', {}).get('test', {})
                if onehot:
                    print(f"    One-hot target:        AUC = {onehot.get('auc_roc', 'N/A'):.3f}")
                print(f"    CRAFT-augmented:       AUC = "
                      f"{t4.get('craft_augmented', {}).get('test', {}).get('auc_roc', 'N/A'):.3f}")
                craft_vs_base = d.get('craft_vs_baseline_auc', d.get('test_auc_roc', 0))
                craft_vs_onehot = d.get('craft_vs_onehot_auc')
                improves_base = d.get('craft_improves_over_baseline', d.get('craft_improves', False))
                print(f"    CRAFT vs baseline:     {craft_vs_base:+.3f}"
                      f"  {'[+]' if improves_base else '[-]'}")
                if craft_vs_onehot is not None:
                    improves_onehot = d.get('craft_improves_over_onehot', False)
                    print(f"    CRAFT vs one-hot:      {craft_vs_onehot:+.3f}"
                          f"  {'[+]' if improves_onehot else '[-]'}")

        if 'task5' in results:
            t5 = results['task5']
            print(f"\n  Task 5: Standard Tox21 (Per-Task Benchmarking)")
            if _is_multi_seed(t5):
                agg = t5.get('aggregated', {})
                print(f"    Mean AUC:              {_fmt_ci(agg.get('mean_auc'))}")
            else:
                mean_auc = t5.get('mean_auc', 'N/A')
                if isinstance(mean_auc, (int, float)):
                    print(f"    Mean AUC:              {mean_auc:.3f}")

        if 'task6' in results:
            t6 = results['task6']
            print(f"\n  Task 6: LOFO Generalization")
            per_family = t6.get('per_family', {})
            if _is_multi_seed(t6):
                per_family = t6.get('aggregated', {}).get('per_family', per_family)
            for fam, metrics in per_family.items():
                if isinstance(metrics, dict):
                    print(f"    {fam}: CRAFT R²={metrics.get('craft_r2', 'N/A')}")

        print("\n" + "=" * 70)
