# CRAFT: A 256-Bit Biological Fingerprint for Drug Targets

**Context-Rich Annotated Fingerprint for Targets**

This repository contains the CRAFT fingerprint schema, encoding engine, and
validation pipeline accompanying the manuscript:

> Jebril I.H., Alshehade S.A. et al. "CRAFT: A 256-Bit Biological Fingerprint
> for Drug Targets Encoding Subcellular Context, Signaling Mechanism, and
> Binding Pocket Architecture for Multi-Target Drug Discovery."
> *Journal of Biomedical Informatics* (2026, under review).

---

## Contents

```
craft/
  schema.py          256-bit schema definition (all bit names, descriptions, sources)
  encoder.py         CRAFT encoding engine (all four modules + API clients)

validation/
  task1_binding_affinity.py    Multi-target pIC50 regression (10 targets, 17,769 pairs)
  task2_tox21_pooled.py        Pooled long-format Tox21 classification (12 assays)
  task3_selectivity.py         Tanimoto selectivity heatmap (20 targets)
  task4_polypharmacology.py    Polypharmacology binary classification
  task5_tox21_standard.py      Standard Tox21 benchmark (one classifier per assay)
  task6_lofo.py                Leave-one-family-out binding affinity generalization
  utils.py                     Shared helpers (scaffold split, metrics, caching)

data/
  tox21_target_mapping.json    12 assay -> UniProt mapping (Table S2)
  selectivity_panel.json       20-target panel (Table S3)
  lofo_target_panel.json       50-target panel (Table S8)

results/june_2026/
  task1_seed_*/metrics.json    Per-seed binding affinity metrics
  task2_seed_*/metrics.json    Per-seed Tox21 metrics
  task3_selectivity.json       Selectivity matrix + Mann-Whitney results
  task4_seed_*/metrics.json    Per-seed polypharmacology metrics
  task5_tox21_per_assay.json   Standard Tox21 per-assay AUC (Table S6)
  task6_lofo.json              LOFO R2 by encoding x family (Table S7)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+. Key dependencies: `rdkit`, `scikit-learn`, `deepchem`,
`requests`, `numpy`, `scipy`.

---

## Quick Start

### Generate a CRAFT fingerprint

```python
from craft.encoder import CRAFTEncoder

encoder = CRAFTEncoder()

# Input: UniProt ID, gene name, or ChEMBL ID
fingerprint = encoder.encode("P07550")      # ADRB2 (beta-2 adrenergic receptor)
fingerprint = encoder.encode("EGFR")        # by gene name
fingerprint = encoder.encode("CHEMBL203")   # by ChEMBL ID

print(f"CRAFT vector shape: {fingerprint.vector.shape}")  # (256,)
print(f"ON bits: {fingerprint.on_bit_count}")
print(f"Module A bits: {fingerprint.module_a.sum()}")
```

### Fuse CRAFT with ECFP4

```python
from craft.encoder import CRAFTEncoder
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

encoder = CRAFTEncoder()
craft_vec = encoder.encode("EGFR").vector   # (256,) numpy array

mol = Chem.MolFromSmiles("c1ccc(Nc2ncnc3c2ccc(OCCCN4CCCCC4)c3)cc1")
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ecfp4 = gen.GetFingerprintAsNumPy(mol)     # (2048,)

import numpy as np
fused = np.concatenate([ecfp4, craft_vec]) # (2304,)
```

### Reproduce all validation tasks

```bash
python -m validation.task1_binding_affinity --seeds 42 123 456 789 2024
python -m validation.task2_tox21_pooled     --seeds 42 123 456 789 2024
python -m validation.task3_selectivity
python -m validation.task4_polypharmacology --seeds 42 123 456 789 2024
python -m validation.task5_tox21_standard   --seeds 42 123 456 789 2024
python -m validation.task6_lofo             --seeds 42 123 456 789 2024
```

API responses and ESM-2 embeddings are disk-cached (7-day TTL) under
`.craft_cache/` for deterministic re-runs.

---

## Schema Overview

| Module | Bits   | Biological Dimension        | Primary Source  |
|--------|--------|-----------------------------|-----------------|
| A      | 0-63   | Membrane topology + localization | UniProt     |
| B      | 64-127 | Signal transduction mechanism    | IUPHAR/BPS  |
| C      | 128-191| Binding pocket character         | PDB / PDBe  |
| D      | 192-255| Endogenous ligand type           | IUPHAR/BPS  |

Each module contains 4 sub-blocks of 16 bits. Full schema with biological
justifications for all 256 positions is in Supplementary Table S1 of the
paper and in `craft/schema.py`.

---

## Citation

If you use CRAFT in your research, please cite:

```
Jebril I.H., Alshehade S.A., et al. CRAFT: A 256-Bit Biological Fingerprint
for Drug Targets. Journal of Biomedical Informatics (2026). [under review]
https://github.com/salahalsh/craft
```

---

## License

MIT License. Data from UniProt, IUPHAR/BPS, ChEMBL, and PDB are subject to
their respective open-access terms.

---

## Contact

Salah A. Alshehade - salah_alsh@outlook.com
ORCID: 0000-0001-8732-1883
