# CRAFT: Context-Rich Annotated Fingerprint for Targets

CRAFT is a 256-bit binary fingerprint for drug targets. Every bit is defined
in a schema (`craft/services/craft_schema.py`) with a name, a biological
justification and the public database it comes from. This repository holds
the encoder and the validation framework that produced every result in the
accompanying manuscript, together with the API response cache those results
were computed from, so that the validation can be repeated offline.

Version 0.3.0. Released under the MIT licence (see `LICENSE`).

## Layout

```
craft/
  services/            encoding engine
    craft_schema.py      the 256 bit definitions (source of Supplementary Table S1)
    craft_generator.py   identifier -> 256-bit fingerprint (four modules)
    chembl_client.py     identifier resolution (ChEMBL)
    uniprot_client.py    localization, topology, tissue, keywords (Module A)
    iuphar_client.py     signalling and endogenous ligands (Modules B and D)
    pdbe_client.py       binding-site residues of one selected structure
    pocket_analyzer.py   Module C, three-tier fallback
    craft_similarity.py  Tanimoto and per-module similarity
  training/            validation framework
    target_panels.py     all target panels and the Tox21 assay-to-target map
    bioactivity_fetcher.py, dataset_builders.py, featurizers.py,
    baseline_featurizers.py, validation_tasks.py, statistical_utils.py
  data/cache/          API response cache used for the published run
  config.py            environment-variable settings (below)
  validate.py          command-line entry point (python -m craft.validate)
  ablation_tox21.py    Tox21 logistic-regression ablation
tests/                 offline regression tests
porting/               the script that extracted this package from the platform,
                       and the original platform command it replaces
```

The code was developed inside the Insilico Sigma web platform and extracted
with the web-framework layer replaced: `config.py` stands in for the platform
settings and `validate.py` for its `validate_craft` management command
(`porting/` holds both, so the extraction can be checked). Task construction,
models and statistics are unchanged.

This repository previously held a separate standalone encoder (release v0.1).
It is replaced by the code above, which is the code used for the validation.

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the exact versions used for the published results
(Python 3.13 on Windows 11, the only platform on which the release has been
tested). DeepChem is needed only for the Tox21 tasks, and PyTorch and
Transformers only for the ESM-2 control. Run all commands from the repository
root.

## Configuration

| Variable | Meaning | Default |
|---|---|---|
| `CRAFT_CACHE_DIR` | root of the API response caches | `craft/data/cache` |
| `CRAFT_CACHE_TTL_HOURS` | cache lifetime in hours; `0` = never expire | `168` |
| `CRAFT_DISABLE_IUPHAR` | `1` = do not query IUPHAR/BPS for any target | unset |

## Generating a fingerprint

```python
from craft.services.craft_generator import CRAFTGenerator
from craft.services.craft_similarity import CRAFTSimilarity

gen = CRAFTGenerator()
egfr = gen.generate("P00533")         # UniProt accession recommended
adrb2 = gen.generate("P07550")
print(egfr.total_on_bits, egfr.module_c_tier, egfr.data_coverage)
print(CRAFTSimilarity().tanimoto(egfr.vector, adrb2.vector))
print(egfr.to_dict()["bit_annotations"])   # name, source and basis of each active bit
```

New targets need network access to ChEMBL, UniProt and PDBe. Each
fingerprint records which sources returned data (`data_coverage`), any
warnings, and the Module C tier.

## Reproducing the validation offline

The validation is run with IUPHAR disabled (see below) and from the cache
shipped in `craft/data/cache`. To use that cache as it is:

```bash
# Linux / macOS
export CRAFT_DISABLE_IUPHAR=1 CRAFT_CACHE_TTL_HOURS=0
# Windows PowerShell
$env:CRAFT_DISABLE_IUPHAR="1"; $env:CRAFT_CACHE_TTL_HOURS="0"

# all six tasks, five seeds, with the identity controls (random per-target
# vector, ESM-2, metadata) wherever a task supports them
python -m craft.validate --task all --multi-seed --include-baselines --output-dir results/

# Tox21 logistic-regression ablation (ECFP4 alone, CRAFT alone, both)
python -m craft.ablation_tox21 results/ablation_tox21
```

Each task writes `metrics.json` (per-seed metrics and aggregates) and its
test-set predictions to its own subdirectory. The DeepChem loader downloads
the Tox21 data file on first use. A full run takes well over a day on a
desktop CPU, most of it Task 6; a single task can be run with `--task` and a
single seed with `--seeds 42`.

## Tests

```bash
python -m pytest tests      # or: python tests/test_regressions.py
```

The tests need no network access. They cover the v0.3.0 fixes below and check
that every bit position an encoder writes lies in its own module and is
declared in the schema.

## Changes in v0.3.0

v0.2.0 (not released) was the code of a first validation run. Auditing it
found that several data fields were silently empty for every target. All are
fixed here, and every fingerprint changes as a result:

- **UniProt localization and tissue specificity were never parsed.** The REST
  JSON writes comment types with a space (`SUBCELLULAR LOCATION`,
  `TISSUE SPECIFICITY`); the parser compared with underscores, so sub-block A1
  (localization) was empty for every target and A3 (tissue) used GO terms only.
  The UniProt cache key is versioned so old parsed records are not reused.
- **The target class was always empty.** ChEMBL `/target` records have no class
  field. The class is now read from ChEMBL's protein classification
  (`/target_component` and `/protein_classification`), which fills the family
  sub-block A4 and the class-derived Module B bits.
- **Module C Tier 2 was unreachable.** `PocketAnalyzer._match_class` returned
  before its keyword fallback when the class was empty; GPCR class names were
  not recognised. Targets without a usable X-ray structure now get the
  class-based pocket defaults.
- **Family keyword matches were not word-bounded.** `ras` matched inside
  `transferase` and `isomerase`, `hsp` inside other words.
- **The metadata control returned one vector for every target.** It called a
  UniProt method that does not exist (logged at debug level, now a warning)
  and its family and compartment terms never matched the real vocabulary.
- **Gene-symbol resolution** now prefers a human single-protein ChEMBL target
  with an exact symbol match (v0.2.0 resolved `EGFR` to an EGFR/PPP1CA
  protein-protein-interaction target).

## Remaining limitations

- **IUPHAR/BPS requires an API key.** The Guide to Pharmacology web services
  reject anonymous requests (HTTP 401), so validation runs use
  `CRAFT_DISABLE_IUPHAR=1`; Module D and the IUPHAR-derived parts of Module B
  stay empty.
- **Module C analyses one structure.** Of the 15 best-resolved X-ray
  structures at 4.0 A or better, the best-resolved one with a drug-like ligand,
  otherwise the best-resolved one; cryo-EM and NMR structures are not used.
- **Annotation is keyword-matched**, which will miss non-standard vocabulary.
- **Fingerprints are not guaranteed unique**; of the 62 validation targets,
  CHRM1 and CHRM2 are identical.

## Data sources

ChEMBL, UniProt, PDBe and (when enabled) the IUPHAR/BPS Guide to
Pharmacology; Tox21 through DeepChem/MoleculeNet. These retain their own
licences. The cached responses were retrieved between 29 September and
3 October 2026.
