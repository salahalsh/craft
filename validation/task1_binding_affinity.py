"""
Task 1: Multi-target binding affinity prediction.

Pools compound-target pairs from 10 diverse protein targets across four
protein families. Compares ECFP4-only vs. ECFP4 + CRAFT (2304-bit fused).

Usage:
    python -m validation.task1_binding_affinity
    python -m validation.task1_binding_affinity --seeds 42 123 456 789 2024
    python -m validation.task1_binding_affinity --output results/june_2026/task1

Reference:
    Jebril I.H., Alshehade S.A. et al.
    "CRAFT: A 256-Bit Biological Fingerprint for Drug Targets."
    Journal of Biomedical Informatics (2026). Table 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from craft.encoder import CRAFTEncoder
from validation.utils import (
    scaffold_split, rf_gb_regressor,
    ensemble_predict_regression, evaluate_regression,
    multi_seed_run, wilcoxon_paired, SEEDS,
)

# ---------------------------------------------------------------------------
# Target panel (Task 1)  --  10 diverse protein targets, 4 families
# ---------------------------------------------------------------------------

TARGETS = [
    {"gene": "EGFR",   "uniprot": "P00533", "family": "RTK"},
    {"gene": "ERBB2",  "uniprot": "P04626", "family": "RTK"},
    {"gene": "MAPK14", "uniprot": "Q16539", "family": "Kinase"},
    {"gene": "SRC",    "uniprot": "P12931", "family": "Kinase"},
    {"gene": "MAPK1",  "uniprot": "P28482", "family": "Kinase"},
    {"gene": "CA2",    "uniprot": "P00918", "family": "Enzyme"},
    {"gene": "ADRB2",  "uniprot": "P07550", "family": "GPCR"},
    {"gene": "PTGS2",  "uniprot": "P35354", "family": "Enzyme"},
    {"gene": "ESR1",   "uniprot": "P03372", "family": "Nuclear Receptor"},
    {"gene": "ALOX5",  "uniprot": "P09917", "family": "Enzyme"},
]

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
ECFP_SIZE  = 2048
ECFP_RADIUS = 2


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _pic50(val_nm: float) -> float:
    """Convert bioactivity value (nM) to pIC50."""
    return 9.0 - np.log10(val_nm)


def fetch_bioactivity(uniprot_id: str, max_activity_nm: float = 50_000) -> list[dict]:
    """Fetch IC50/Ki data from ChEMBL for a single target."""
    results = []
    offset = 0
    limit = 1000
    while True:
        url = f"{CHEMBL_API}/activity.json"
        params = {
            "target_chembl_id__target_components__accession": uniprot_id,
            "standard_type__in": "IC50,Ki",
            "standard_relation": "=",
            "standard_units": "nM",
            "limit": limit,
            "offset": offset,
        }
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
        except Exception:
            break

        activities = data.get("activities", [])
        if not activities:
            break

        for act in activities:
            val = act.get("standard_value")
            smi = act.get("canonical_smiles") or act.get("molecule_structures", {}).get("canonical_smiles")
            if val is None or smi is None:
                continue
            try:
                val = float(val)
            except (ValueError, TypeError):
                continue
            if val <= 0 or val > max_activity_nm:
                continue
            results.append({"smiles": smi, "value_nm": val, "pic50": _pic50(val)})

        if len(activities) < limit:
            break
        offset += limit

    return results


def load_task1_dataset(cache_file: Path | None = None) -> tuple[list, np.ndarray, list]:
    """
    Load and pool all compound-target pairs for Task 1.

    Returns
    -------
    smiles_list  : list[str]   - SMILES strings (deduplicated per target)
    labels       : np.ndarray  - pIC50 values, shape (N,)
    target_ids   : list[str]   - UniProt ID for each row (for CRAFT tiling)
    """
    if cache_file and cache_file.exists():
        with open(cache_file) as f:
            d = json.load(f)
        return d["smiles"], np.array(d["labels"]), d["target_ids"]

    encoder = CRAFTEncoder()
    all_smiles, all_labels, all_target_ids = [], [], []

    for target in TARGETS:
        uid = target["uniprot"]
        records = fetch_bioactivity(uid)

        # Deduplicate: geometric mean for same SMILES
        smi_to_vals: dict[str, list[float]] = {}
        for rec in records:
            smi = rec["smiles"]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            smi_to_vals.setdefault(canonical, []).append(rec["value_nm"])

        for smi, vals in smi_to_vals.items():
            geo_mean = float(np.exp(np.mean(np.log(vals))))
            all_smiles.append(smi)
            all_labels.append(_pic50(geo_mean))
            all_target_ids.append(uid)

    labels = np.array(all_labels)

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({"smiles": all_smiles, "labels": labels.tolist(),
                       "target_ids": all_target_ids}, f)

    return all_smiles, labels, all_target_ids


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

_ecfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP_RADIUS, fpSize=ECFP_SIZE)
_craft_encoder = CRAFTEncoder()
_craft_cache: dict[str, np.ndarray] = {}


def ecfp4(smiles_list: list[str]) -> np.ndarray:
    X = np.zeros((len(smiles_list), ECFP_SIZE), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            X[i] = _ecfp_gen.GetFingerprintAsNumPy(mol)
    return X


def craft_tile(target_ids: list[str]) -> np.ndarray:
    """Return (N, 256) array of CRAFT vectors tiled per target."""
    out = np.zeros((len(target_ids), 256), dtype=np.uint8)
    for i, uid in enumerate(target_ids):
        if uid not in _craft_cache:
            _craft_cache[uid] = _craft_encoder.encode(uid).vector
        out[i] = _craft_cache[uid]
    return out


# ---------------------------------------------------------------------------
# Single-seed experiment
# ---------------------------------------------------------------------------

def run_task1_seed(smiles: list, labels: np.ndarray, target_ids: list,
                   seed: int) -> dict[str, float]:
    X_chem = ecfp4(smiles)
    X_craft_vec = craft_tile(target_ids)
    X_fused = np.concatenate([X_chem, X_craft_vec], axis=1)

    tr, val, te = scaffold_split(smiles, labels, seed=seed)

    results = {}
    for condition, X in [("baseline", X_chem), ("craft", X_fused)]:
        rf, gb = rf_gb_regressor(seed=seed)
        X_tr, y_tr = X[tr], labels[tr]
        X_te, y_te = X[te], labels[te]
        rf.fit(X_tr, y_tr)
        gb.fit(X_tr, y_tr)
        y_pred = ensemble_predict_regression(rf, gb, X_te)
        metrics = evaluate_regression(y_te, y_pred)
        for k, v in metrics.items():
            results[f"{condition}_{k}"] = v

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--output", type=Path, default=Path("results/task1"))
    parser.add_argument("--cache",  type=Path, default=Path(".craft_cache/task1_dataset.json"))
    args = parser.parse_args()

    print("Loading Task 1 dataset...")
    smiles, labels, target_ids = load_task1_dataset(cache_file=args.cache)
    print(f"  {len(smiles)} compound-target pairs loaded.")

    def experiment(seed: int) -> dict[str, float]:
        return run_task1_seed(smiles, labels, target_ids, seed)

    print(f"Running Task 1 with seeds {args.seeds}...")
    summary = multi_seed_run(experiment, seeds=args.seeds)

    baseline_r2 = [summary["baseline_r2"]["values"][i] for i in range(len(args.seeds))]
    craft_r2    = [summary["craft_r2"]["values"][i]    for i in range(len(args.seeds))]
    p_val = wilcoxon_paired(baseline_r2, craft_r2)

    print("\n=== Task 1: Multi-target Binding Affinity ===")
    print(f"  Baseline R2: {summary['baseline_r2']['mean']:.3f} +/- "
          f"{summary['baseline_r2']['std']:.3f} "
          f"[{summary['baseline_r2']['ci_low']:.3f}, {summary['baseline_r2']['ci_high']:.3f}]")
    print(f"  CRAFT    R2: {summary['craft_r2']['mean']:.3f} +/- "
          f"{summary['craft_r2']['std']:.3f} "
          f"[{summary['craft_r2']['ci_low']:.3f}, {summary['craft_r2']['ci_high']:.3f}]")
    print(f"  Delta R2   : {summary['craft_r2']['mean'] - summary['baseline_r2']['mean']:+.3f}")
    print(f"  Wilcoxon p : {p_val:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    with open(args.output / "task1_summary.json", "w") as f:
        json.dump({"summary": summary, "wilcoxon_p": p_val,
                   "n_pairs": len(smiles), "seeds": args.seeds}, f, indent=2)
    print(f"\nResults saved to {args.output}/task1_summary.json")


if __name__ == "__main__":
    main()
