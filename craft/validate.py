"""Run the CRAFT validation framework (Tasks 1-6).

Standalone port of the Insilico Sigma management command `validate_craft`
(see _original_validate_craft_command.py): same options, same task order,
same task construction. Only the Django command wrapper is replaced.

Usage:
    python -m craft.validate --task all --multi-seed --output-dir results/
    python -m craft.validate --task affinity --multi-seed --include-baselines

Reproducing the published results offline (see README):
    CRAFT_DISABLE_IUPHAR=1 CRAFT_CACHE_TTL_HOURS=0 \
        python -m craft.validate --task all --multi-seed --output-dir results/
"""
import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m craft.validate",
        description="Run CRAFT validation framework - demonstrate CRAFT "
                    "improves ML predictions")
    p.add_argument("--task", choices=["affinity", "admet", "selectivity",
                                      "polypharmacology", "standard_tox21",
                                      "lofo", "all"], default="all",
                   help="Which validation task(s) to run (default: all)")
    p.add_argument("--max-targets", type=int, default=0,
                   help="Limit number of targets for quick testing (0 = use all)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for results (default: ./craft_results)")
    p.add_argument("--skip-fetch", action="store_true",
                   help="Skip API fetching - use cached data only")
    p.add_argument("--multi-seed", action="store_true",
                   help="Run each task with 5 random seeds and report mean +/- CI")
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds (default: 42,123,456,789,2024)")
    p.add_argument("--include-baselines", action="store_true",
                   help="Include competitive baselines (random, ESM-2, metadata)")
    p.add_argument("--use-extended", action="store_true",
                   help="Use extended 50-target panel instead of original 10")
    p.add_argument("--n-jobs", type=int, default=-1,
                   help="Parallel jobs for sklearn. -1 = all cores (default), "
                        "0 = 80%% of cores, or an exact count")
    return p.parse_args(argv)


def _check_dependencies(task: str):
    """Verify required packages are available."""
    errors = []
    try:
        import numpy
        print(f"  numpy: {numpy.__version__}")
    except ImportError:
        errors.append("numpy")
    try:
        import sklearn
        print(f"  scikit-learn: {sklearn.__version__}")
    except ImportError:
        errors.append("scikit-learn")
    try:
        from rdkit import rdBase
        print(f"  RDKit: {rdBase.rdkitVersion}")
    except ImportError:
        errors.append("RDKit")
    if task in ("admet", "all"):
        try:
            import deepchem as dc
            print(f"  DeepChem: {dc.__version__}")
        except ImportError:
            errors.append("DeepChem (needed for --task admet)")
    if task in ("selectivity", "all"):
        try:
            from scipy import stats  # noqa: F401
            print("  scipy: available")
        except ImportError:
            errors.append("scipy (needed for --task selectivity)")
    if errors:
        print(f"Missing dependencies: {', '.join(errors)}")
        raise SystemExit(1)
    print("")


def main(argv=None):
    options = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    task = options.task
    max_targets = options.max_targets
    multi_seed = options.multi_seed
    include_baselines = options.include_baselines
    use_extended = options.use_extended

    # Resolve n_jobs: -1=all, 0=80%, positive=exact
    if options.n_jobs == 0:
        cpu_count = os.cpu_count() or 4
        n_jobs = max(1, int(cpu_count * 0.8))
        print(f"  n_jobs: {n_jobs} (80% of {cpu_count} cores)")
    else:
        n_jobs = options.n_jobs

    seeds = None
    if options.seeds:
        seeds = [int(s.strip()) for s in options.seeds.split(",")]
    elif multi_seed:
        from craft.training.statistical_utils import SEEDS
        seeds = SEEDS

    output_dir = options.output_dir or str(Path.cwd() / "craft_results")

    seed_info = f" | Seeds: {seeds}" if multi_seed else ""
    print(f"\n{'=' * 60}\n  CRAFT Validation Framework\n"
          f"  Task: {task} | Max targets: {max_targets or 'all'}{seed_info}\n"
          f"  Output: {output_dir}\n{'=' * 60}\n")

    _check_dependencies(task)

    from craft.training.validation_tasks import (
        BindingAffinityTask, ADMETTask, SelectivityTask, PolypharmacologyTask)
    from craft.training.validation_report import ValidationReport

    results = {}

    def _run_task(task_obj, task_key, task_label):
        """Run a task with optional multi-seed."""
        try:
            if multi_seed:
                results[task_key] = task_obj.run_multi_seed(seeds)
            else:
                results[task_key] = task_obj.run()
            print(f"{task_label} complete" + (" (multi-seed)" if multi_seed else ""))
        except Exception as e:
            print(f"{task_label} failed: {e}")
            logger.exception(f"{task_label} error")

    # Task 3 first - fastest (no ML, no API fetching for bioactivity)
    if task in ("selectivity", "all"):
        task3 = SelectivityTask(
            output_dir=os.path.join(output_dir, "task3_selectivity"),
            max_targets=max_targets,
            n_jobs=n_jobs,
        )
        _run_task(task3, "task3", "Task 3 (Selectivity)")

    # Task 1 - Binding affinity (regression)
    if task in ("affinity", "all"):
        task1 = BindingAffinityTask(
            output_dir=os.path.join(output_dir, "task1_binding_affinity"),
            max_targets=max_targets,
            include_baselines=include_baselines,
            use_extended=use_extended,
            n_jobs=n_jobs,
        )
        _run_task(task1, "task1", "Task 1 (Binding Affinity)")

    # Task 2 - ADMET (Tox21)
    if task in ("admet", "all"):
        task2 = ADMETTask(
            output_dir=os.path.join(output_dir, "task2_admet"),
            max_targets=max_targets,
            include_baselines=include_baselines,
            n_jobs=n_jobs,
        )
        _run_task(task2, "task2", "Task 2 (ADMET)")

    # Task 4 - Polypharmacology
    if task in ("polypharmacology", "all"):
        task4 = PolypharmacologyTask(
            output_dir=os.path.join(output_dir, "task4_polypharmacology"),
            max_targets=max_targets,
            include_baselines=include_baselines,
            use_extended=use_extended,
            n_jobs=n_jobs,
        )
        _run_task(task4, "task4", "Task 4 (Polypharmacology)")

    # Task 5 - Standard Tox21 (per-task benchmarking)
    if task in ("standard_tox21", "all"):
        from craft.training.validation_tasks import StandardTox21Task
        task5 = StandardTox21Task(
            output_dir=os.path.join(output_dir, "task5_standard_tox21"),
            max_targets=max_targets,
            n_jobs=n_jobs,
        )
        _run_task(task5, "task5", "Task 5 (Standard Tox21)")

    # Task 6 - LOFO generalization (smaller ensembles, as in the published run)
    if task in ("lofo", "all"):
        from craft.training.validation_tasks import LOFOTask
        task6 = LOFOTask(
            output_dir=os.path.join(output_dir, "task6_lofo"),
            max_targets=max_targets,
            n_jobs=n_jobs,
            rf_n_estimators=200,
            gb_n_estimators=100,
        )
        _run_task(task6, "task6", "Task 6 (LOFO Generalization)")

    if results:
        report = ValidationReport()
        summary_path = os.path.join(output_dir, "validation_summary.json")
        report.generate(results, summary_path)
        report.print_summary(results)
        print(f"\nValidation summary saved to: {summary_path}")
    else:
        print("No tasks completed successfully.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
