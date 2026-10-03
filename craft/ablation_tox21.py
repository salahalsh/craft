"""Tox21 logistic-regression ablation: ECFP4 alone, CRAFT alone, ECFP4 + CRAFT.

Pooled Tox21 (measured compound-assay pairs only), the five standard seeds and
the same scaffold split as the validation tasks. Because a CRAFT vector is
constant within an assay, the CRAFT-only model can assign one score per
distinct vector, so its AUC measures how much the assays differ in their
positive rates.

Usage: python -m craft.ablation_tox21 <output_dir>
"""
import json
import logging
import os
import sys
import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EXPECTED_ROWS = 77864  # measured pairs in the MoleculeNet Tox21 release


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.exit(__doc__)
    out = os.path.abspath(argv[0])
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    from craft.training.dataset_builders import ADMETDatasetBuilder
    from craft.training.statistical_utils import SEEDS
    from craft.training.validation_tasks import BaseValidationTask

    print("Building ADMET dataset...", flush=True)
    t0 = time.time()
    data = ADMETDatasetBuilder().build()
    print(f"  built in {time.time() - t0:.0f}s", flush=True)

    X_ecfp, X_aug, y, smiles = (data["X_ecfp"], data["X_craft_augmented"],
                                data["y"], data["smiles"])
    X_craft = X_aug[:, 2048:]
    n_unique = len({tuple(r) for r in X_craft.astype(int)})
    print(f"Dataset: {len(y)} rows, {int(y.sum())} positive ({100 * y.mean():.2f}%); "
          f"{n_unique} distinct CRAFT vectors", flush=True)
    if len(y) != EXPECTED_ROWS:
        sys.exit(f"expected {EXPECTED_ROWS} measured rows, got {len(y)}")
    if n_unique < 2:
        sys.exit("CRAFT-only features are constant; fingerprint generation failed")

    os.makedirs(out, exist_ok=True)
    task = BaseValidationTask(output_dir=out)
    results = {"ecfp_only": [], "craft_only": [], "ecfp_craft": []}
    for seed in SEEDS:
        task.seed = seed
        train_idx, _val_idx, test_idx = task._scaffold_split(smiles)
        for cond, X in (("craft_only", X_craft), ("ecfp_only", X_ecfp), ("ecfp_craft", X_aug)):
            lr = LogisticRegression(max_iter=2000, class_weight="balanced",
                                    random_state=seed, C=1.0)
            lr.fit(X[train_idx], y[train_idx])
            p = lr.predict_proba(X[test_idx])[:, 1]
            r = {"seed": seed, "auc": float(roc_auc_score(y[test_idx], p)),
                 "auprc": float(average_precision_score(y[test_idx], p))}
            results[cond].append(r)
            print(f"  seed {seed} {cond:11s} AUC={r['auc']:.3f} AUPRC={r['auprc']:.3f}", flush=True)

    summary = {cond: {"auc_mean": float(np.mean([r["auc"] for r in rows])),
                      "auc_sd": float(np.std([r["auc"] for r in rows], ddof=1)),
                      "auprc_mean": float(np.mean([r["auprc"] for r in rows]))}
               for cond, rows in results.items()}
    for cond, sm in summary.items():
        print(f"{cond:11s}: AUC {sm['auc_mean']:.3f} +/- {sm['auc_sd']:.3f}")
    with open(os.path.join(out, "ablation_results_5seed.json"), "w") as f:
        json.dump({"seeds": list(SEEDS), "n_rows": int(len(y)),
                   "n_unique_craft_vectors": n_unique,
                   "per_seed": results, "summary": summary}, f, indent=2)
    print(f"Saved {out}/ablation_results_5seed.json")


if __name__ == "__main__":
    main()
