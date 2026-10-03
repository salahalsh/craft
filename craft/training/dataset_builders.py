"""
Dataset Builders for CRAFT Validation Tasks.

Each builder constructs a specific dataset needed by the validation tasks:
- BindingAffinityDatasetBuilder: Multi-target pIC50 regression (Task 1)
- ADMETDatasetBuilder: Pooled Tox21 multi-task classification (Task 2)
- SelectivityDatasetBuilder: CRAFT similarity matrix (Task 3)
- PolypharmacologyDatasetBuilder: Cross-target binary classification (Task 4)
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _require_craft_vector(featurizer, identifier, label):
    """Generate a CRAFT vector for a panel target, or fail loudly.

    Every panel in this module is curated. A target whose fingerprint cannot be
    generated must stop the run rather than be skipped with a warning, because
    a silently shrunken panel changes every reported number while looking like
    a successful run.
    """
    vec = featurizer.generate_craft_vector(identifier)
    if vec is None:
        raise RuntimeError(
            f"CRAFT generation failed for {label} ({identifier}); refusing to "
            f"silently drop a panel target")
    return vec


class BindingAffinityDatasetBuilder:
    """
    Build multi-target binding affinity dataset (Task 1).

    Fetches IC50/Ki data from ChEMBL for multiple targets, generates ECFP4
    for compounds and CRAFT for targets, and pools into a single dataset.
    """

    def build(self, targets: List[Dict], max_per_target: int = 2000,
              max_value_nm: float = 50000, include_baselines: bool = False
              ) -> Dict:
        """
        Build the multi-target binding affinity dataset.

        Args:
            targets: List of target dicts with 'chembl_id' and 'uniprot_id'
            max_per_target: Max compounds per target
            max_value_nm: Max IC50/Ki in nM (default 50μM)

        Returns:
            Dict with keys:
                'X_ecfp': np.ndarray (n, 2048)
                'X_craft_augmented': np.ndarray (n, 2304)
                'y': np.ndarray (n,) - pIC50 values
                'smiles': List[str]
                'target_ids': List[str] - chembl_id per sample
                'target_genes': List[str] - gene name per sample
                'metadata': dict
        """
        from .bioactivity_fetcher import ChEMBLBioactivityFetcher
        from .featurizers import CRAFTTrainingFeaturizer

        fetcher = ChEMBLBioactivityFetcher()
        featurizer = CRAFTTrainingFeaturizer()

        all_smiles = []
        all_pIC50 = []
        all_target_ids = []
        all_target_genes = []
        craft_vectors_per_sample = []
        per_target_counts = {}

        # Pre-generate CRAFT vectors for all targets
        logger.info("Generating CRAFT vectors for %d targets...", len(targets))
        craft_cache = {}
        for t in targets:
            ident = t.get('uniprot_id') or t.get('chembl_id')
            craft_cache[t['chembl_id']] = _require_craft_vector(
                featurizer, ident, t.get('gene', ident))

        # Fetch bioactivities per target
        for t in targets:
            chembl_id = t['chembl_id']
            if chembl_id not in craft_cache:
                continue

            craft_vec = craft_cache[chembl_id]

            logger.info("Fetching bioactivities for %s (%s)...",
                        t.get('gene', ''), chembl_id)
            records = fetcher.fetch_bioactivities(
                chembl_id, max_value_nm=max_value_nm, limit=max_per_target)

            if not records:
                logger.warning("No bioactivity data for %s", chembl_id)
                continue

            for r in records:
                all_smiles.append(r.smiles)
                all_pIC50.append(r.pIC50)
                all_target_ids.append(chembl_id)
                all_target_genes.append(t.get('gene', ''))
                craft_vectors_per_sample.append(craft_vec)

            per_target_counts[t.get('gene', chembl_id)] = len(records)
            logger.info("  %s: %d compounds", t.get('gene', ''), len(records))

        if not all_smiles:
            raise ValueError("No bioactivity data fetched for any target")

        # Generate ECFP4 for all compounds
        logger.info("Generating ECFP4 for %d compounds...", len(all_smiles))
        X_ecfp = featurizer.featurize_ecfp(all_smiles)

        # Build CRAFT matrix (n, 256) - each row is that sample's target CRAFT
        craft_matrix = np.array(craft_vectors_per_sample, dtype=np.float32)

        # Fused: ECFP4 (2048) + CRAFT (256) = 2304
        X_craft_augmented = featurizer.fuse_ecfp_craft(X_ecfp, craft_matrix)

        y = np.array(all_pIC50, dtype=np.float32)

        logger.info("Dataset built: %d samples, %d targets, pIC50 range [%.1f, %.1f]",
                     len(y), len(per_target_counts), y.min(), y.max())

        result = {
            'X_ecfp': X_ecfp,
            'X_craft_augmented': X_craft_augmented,
            'y': y,
            'smiles': all_smiles,
            'target_ids': all_target_ids,
            'target_genes': all_target_genes,
            'metadata': {
                'n_targets': len(per_target_counts),
                'n_samples': len(y),
                'per_target_counts': per_target_counts,
                'pIC50_mean': float(y.mean()),
                'pIC50_std': float(y.std()),
            },
        }

        if include_baselines:
            result.update(self._build_baseline_matrices(
                X_ecfp, all_target_ids, targets))

        return result

    def _build_baseline_matrices(self, X_ecfp, target_ids, targets):
        """Build baseline augmented matrices for ablation studies."""
        from .baseline_featurizers import (
            RandomBitFeaturizer, ESM2Featurizer, MetadataFeaturizer)

        n = X_ecfp.shape[0]
        baselines = {}

        # Random 256-bit ablation
        logger.info("Generating random-bit baseline...")
        random_feat = RandomBitFeaturizer()
        random_cache = random_feat.batch_generate(
            list(set(target_ids)))
        random_matrix = np.zeros((n, 256), dtype=np.float32)
        for i, tid in enumerate(target_ids):
            if tid in random_cache:
                random_matrix[i] = random_cache[tid]
        baselines['X_random_augmented'] = np.hstack(
            [X_ecfp, random_matrix]).astype(np.float32)

        # ESM-2 embeddings
        logger.info("Generating ESM-2 baseline...")
        try:
            esm_feat = ESM2Featurizer()
            uniprot_map = {t['chembl_id']: t['uniprot_id'] for t in targets}
            uniprot_ids = list(set(uniprot_map.values()))
            esm_cache = esm_feat.batch_generate(uniprot_ids)
            # No zero-filling: a target with no embedding would otherwise get
            # an all-zero ESM-2 vector and silently weaken this baseline.
            missing = sorted(u for u in uniprot_ids if u not in esm_cache)
            if missing:
                raise RuntimeError(
                    f"ESM-2 embedding missing for {len(missing)} target(s): "
                    f"{missing}")
            esm_dim = ESM2Featurizer.EMBEDDING_DIM
            esm_matrix = np.zeros((n, esm_dim), dtype=np.float32)
            for i, tid in enumerate(target_ids):
                esm_matrix[i] = esm_cache[uniprot_map[tid]]
            baselines['X_esm2_augmented'] = np.hstack(
                [X_ecfp, esm_matrix]).astype(np.float32)
        except Exception as e:
            logger.error("ESM-2 baseline SKIPPED (this condition will be "
                         "absent from results): %s", e)

        # Metadata features
        logger.info("Generating metadata baseline...")
        try:
            meta_feat = MetadataFeaturizer()
            meta_cache = meta_feat.batch_generate(
                list(set(target_ids)))
            meta_dim = MetadataFeaturizer.FEATURE_DIM
            meta_matrix = np.zeros((n, meta_dim), dtype=np.float32)
            for i, tid in enumerate(target_ids):
                if tid in meta_cache:
                    meta_matrix[i] = meta_cache[tid]
            baselines['X_metadata_augmented'] = np.hstack(
                [X_ecfp, meta_matrix]).astype(np.float32)
        except Exception as e:
            logger.warning("Metadata baseline skipped: %s", e)

        return baselines


class ADMETDatasetBuilder:
    """
    Build pooled ADMET dataset with CRAFT augmentation (Task 2).

    Loads Tox21 via DeepChem, maps each assay to its protein target,
    generates CRAFT vectors, and creates a long-format pooled dataset.

    Critical: Scaffold split must be at compound level to prevent leakage.
    """

    def build(self, include_baselines: bool = False) -> Dict:
        """
        Build the pooled Tox21 dataset.

        Returns:
            Dict with keys:
                'X_ecfp': np.ndarray (n_pairs, 2048)
                'X_craft_augmented': np.ndarray (n_pairs, 2304)
                'y': np.ndarray (n_pairs,) - binary labels
                'smiles': List[str] - per-row compound SMILES
                'unique_smiles': List[str] - unique compounds (for scaffold split)
                'compound_to_rows': dict - maps SMILES to row indices
                'task_names': List[str] - per-row task name
                'metadata': dict
        """
        import deepchem as dc
        from .target_panels import TOX21_TARGET_MAPPING, get_tox21_craft_identifier
        from .featurizers import CRAFTTrainingFeaturizer

        featurizer = CRAFTTrainingFeaturizer()

        # Load Tox21 via DeepChem (data only)
        logger.info("Loading Tox21 dataset via DeepChem...")
        tasks, datasets, transformers = dc.molnet.load_tox21(
            featurizer='ECFP', splitter='scaffold', reload=False)
        train_ds, valid_ds, test_ds = datasets

        # Pool all splits - we'll re-split by compound-level scaffold
        all_smiles_raw = (list(train_ds.ids) + list(valid_ds.ids)
                          + list(test_ds.ids))
        all_y_raw = np.vstack([train_ds.y, valid_ds.y, test_ds.y])
        # DeepChem marks an unmeasured (compound, assay) result with y = 0 and
        # weight w = 0, NOT with NaN. Filtering on NaN alone therefore keeps
        # every unmeasured result as an "inactive". Measured means w > 0.
        all_w_raw = np.vstack([train_ds.w, valid_ds.w, test_ds.w])

        logger.info("Tox21 loaded: %d compounds, %d tasks, %d measured labels "
                    "(%d unmeasured excluded)",
                    len(all_smiles_raw), len(tasks),
                    int((all_w_raw > 0).sum()), int((all_w_raw == 0).sum()))

        # Generate CRAFT vectors for Tox21 assay targets
        logger.info("Generating CRAFT vectors for Tox21 assay targets...")
        craft_per_task = {}
        for task_name in tasks:
            ident = get_tox21_craft_identifier(task_name)
            if not ident:
                raise RuntimeError(f"No target mapping for Tox21 task {task_name}")
            craft_per_task[task_name] = _require_craft_vector(
                featurizer, ident, task_name)

        if not craft_per_task:
            raise ValueError("No CRAFT vectors generated for any Tox21 task")

        # Build long-format pooled dataset: each row = (compound, task)
        # Only include MEASURED entries (w > 0); see the note on all_w_raw.
        row_smiles = []
        row_labels = []
        row_tasks = []
        row_craft = []

        for task_idx, task_name in enumerate(tasks):
            if task_name not in craft_per_task:
                continue

            craft_vec = craft_per_task[task_name]

            for comp_idx in range(len(all_smiles_raw)):
                label = all_y_raw[comp_idx, task_idx]
                if all_w_raw[comp_idx, task_idx] == 0 or np.isnan(label):
                    continue

                row_smiles.append(all_smiles_raw[comp_idx])
                row_labels.append(int(label))
                row_tasks.append(task_name)
                row_craft.append(craft_vec)

        logger.info("Pooled dataset: %d compound-task pairs (from %d tasks)",
                     len(row_smiles), len(craft_per_task))

        # Generate ECFP4 for all rows
        # Note: same compound appears in multiple rows (once per task)
        # We featurize all rows for simplicity (redundant but correct)
        X_ecfp = featurizer.featurize_ecfp(row_smiles)
        craft_matrix = np.array(row_craft, dtype=np.float32)
        X_craft_augmented = featurizer.fuse_ecfp_craft(X_ecfp, craft_matrix)
        y = np.array(row_labels, dtype=np.float32)

        # Build compound-to-rows mapping for compound-level scaffold split
        unique_smiles = sorted(set(row_smiles))
        compound_to_rows = {}
        for i, smi in enumerate(row_smiles):
            if smi not in compound_to_rows:
                compound_to_rows[smi] = []
            compound_to_rows[smi].append(i)

        # Class distribution
        n_pos = int(y.sum())
        n_neg = len(y) - n_pos

        logger.info("Class distribution: %d positive (%.1f%%), %d negative",
                     n_pos, 100 * n_pos / len(y) if len(y) else 0, n_neg)

        result = {
            'X_ecfp': X_ecfp,
            'X_craft_augmented': X_craft_augmented,
            'y': y,
            'smiles': row_smiles,
            'unique_smiles': unique_smiles,
            'compound_to_rows': compound_to_rows,
            'task_names': row_tasks,
            'metadata': {
                'n_pairs': len(y),
                'n_unique_compounds': len(unique_smiles),
                'n_tasks': len(craft_per_task),
                'tasks_with_craft': list(craft_per_task.keys()),
                'n_positive': n_pos,
                'n_negative': n_neg,
            },
        }

        if include_baselines:
            result.update(self._build_admet_baselines(
                X_ecfp, row_tasks, craft_per_task, tasks, TOX21_TARGET_MAPPING))

        return result

    def _build_admet_baselines(self, X_ecfp, row_tasks, craft_per_task,
                                task_names, target_mapping):
        """Build baseline augmented matrices for ADMET ablation."""
        from .baseline_featurizers import (
            RandomBitFeaturizer, ESM2Featurizer, MetadataFeaturizer)

        n = X_ecfp.shape[0]
        baselines = {}

        # Random 256-bit
        logger.info("Generating ADMET random-bit baseline...")
        random_feat = RandomBitFeaturizer()
        random_per_task = {tn: random_feat.generate(tn) for tn in craft_per_task}
        random_matrix = np.zeros((n, 256), dtype=np.float32)
        for i, tn in enumerate(row_tasks):
            if tn in random_per_task:
                random_matrix[i] = random_per_task[tn]
        baselines['X_random_augmented'] = np.hstack(
            [X_ecfp, random_matrix]).astype(np.float32)

        # ESM-2
        logger.info("Generating ADMET ESM-2 baseline...")
        try:
            esm_feat = ESM2Featurizer()
            uid_map = {tn: info['uniprot_id']
                       for tn, info in target_mapping.items()
                       if info.get('uniprot_id')}
            uids = list(set(uid_map.values()))
            esm_cache = esm_feat.batch_generate(uids)
            esm_dim = ESM2Featurizer.EMBEDDING_DIM
            esm_matrix = np.zeros((n, esm_dim), dtype=np.float32)
            for i, tn in enumerate(row_tasks):
                uid = uid_map.get(tn)
                if uid and uid in esm_cache:
                    esm_matrix[i] = esm_cache[uid]
            baselines['X_esm2_augmented'] = np.hstack(
                [X_ecfp, esm_matrix]).astype(np.float32)
        except Exception as e:
            logger.warning("ESM-2 ADMET baseline skipped: %s", e)

        # Metadata
        logger.info("Generating ADMET metadata baseline...")
        try:
            meta_feat = MetadataFeaturizer()
            meta_per_task = {}
            for tn, info in target_mapping.items():
                uid = info.get('uniprot_id')
                if uid:
                    vec = meta_feat.generate(uid)
                    if vec is not None:
                        meta_per_task[tn] = vec
            meta_dim = MetadataFeaturizer.FEATURE_DIM
            meta_matrix = np.zeros((n, meta_dim), dtype=np.float32)
            for i, tn in enumerate(row_tasks):
                if tn in meta_per_task:
                    meta_matrix[i] = meta_per_task[tn]
            baselines['X_metadata_augmented'] = np.hstack(
                [X_ecfp, meta_matrix]).astype(np.float32)
        except Exception as e:
            logger.warning("Metadata ADMET baseline skipped: %s", e)

        return baselines


class SelectivityDatasetBuilder:
    """
    Build CRAFT similarity matrix for selectivity analysis (Task 3).

    No ML - generates CRAFT for 20 targets (10 kinases + 10 GPCRs) and
    computes pairwise Tanimoto similarity with per-module breakdown.
    """

    def build(self) -> Dict:
        """
        Build selectivity analysis dataset.

        Returns:
            Dict with keys:
                'similarity_matrix': np.ndarray (20, 20)
                'module_similarities': Dict[str, np.ndarray] per module
                'target_labels': List[str] - gene names
                'family_labels': List[str] - 'kinase' or 'gpcr'
                'craft_vectors': np.ndarray (20, 256)
                'metadata': dict
        """
        from .target_panels import KINASE_PANEL, GPCR_PANEL
        from .featurizers import CRAFTTrainingFeaturizer
        from craft.services.craft_similarity import CRAFTSimilarity

        featurizer = CRAFTTrainingFeaturizer()
        similarity = CRAFTSimilarity()

        # Combine panels
        all_targets = []
        family_labels = []
        for t in KINASE_PANEL:
            all_targets.append(t)
            family_labels.append('kinase')
        for t in GPCR_PANEL:
            all_targets.append(t)
            family_labels.append('gpcr')

        # Generate CRAFT vectors
        logger.info("Generating CRAFT vectors for %d selectivity targets...",
                     len(all_targets))
        target_labels = []
        vectors = []
        valid_families = []

        for i, t in enumerate(all_targets):
            ident = t.get('uniprot_id') or t.get('gene')
            vectors.append(_require_craft_vector(featurizer, ident, t['gene']))
            target_labels.append(t['gene'])
            valid_families.append(family_labels[i])

        if len(vectors) < 4:
            raise ValueError(f"Only {len(vectors)} targets generated - need ≥4")

        craft_matrix = np.array(vectors, dtype=np.float32)
        n = len(vectors)

        # Compute overall Tanimoto similarity matrix
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                sim_matrix[i, j] = similarity.tanimoto(vectors[i], vectors[j])

        # Per-module similarity
        from craft.services.craft_schema import MODULE_RANGES
        module_sims = {}
        for mod, (start, end) in MODULE_RANGES.items():
            mod_matrix = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(n):
                    vi = vectors[i][start:end + 1]
                    vj = vectors[j][start:end + 1]
                    mod_matrix[i, j] = similarity.tanimoto(vi, vj)
            module_sims[mod] = mod_matrix

        # Compute within/between family stats
        within_sims = []
        between_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                if valid_families[i] == valid_families[j]:
                    within_sims.append(float(sim_matrix[i, j]))
                else:
                    between_sims.append(float(sim_matrix[i, j]))

        logger.info("Similarity matrix: %dx%d", n, n)
        logger.info("Within-family mean: %.3f (n=%d)",
                     np.mean(within_sims) if within_sims else 0, len(within_sims))
        logger.info("Between-family mean: %.3f (n=%d)",
                     np.mean(between_sims) if between_sims else 0, len(between_sims))

        return {
            'similarity_matrix': sim_matrix,
            'module_similarities': module_sims,
            'target_labels': target_labels,
            'family_labels': valid_families,
            'craft_vectors': craft_matrix,
            'within_family_similarities': within_sims,
            'between_family_similarities': between_sims,
            'metadata': {
                'n_targets': n,
                'n_kinases': valid_families.count('kinase'),
                'n_gpcrs': valid_families.count('gpcr'),
                'within_family_mean': float(np.mean(within_sims)) if within_sims else 0,
                'between_family_mean': float(np.mean(between_sims)) if between_sims else 0,
            },
        }


class PolypharmacologyDatasetBuilder:
    """
    Build polypharmacology classification dataset (Task 4).

    Reuses bioactivity data from Task 1, binarizes into active/inactive,
    and creates cross-target negatives for multi-target classification.
    """

    def build(self, targets: List[Dict], max_per_target: int = 1000,
              active_threshold_nm: float = 1000,
              inactive_threshold_nm: float = 10000,
              include_baselines: bool = False) -> Dict:
        """
        Build polypharmacology binary classification dataset.

        Args:
            targets: Target dicts with 'chembl_id' and 'uniprot_id'
            max_per_target: Max compounds per target
            active_threshold_nm: IC50 < this = active (default 1μM)
            inactive_threshold_nm: IC50 > this = inactive (default 10μM)

        Returns:
            Dict with same structure as BindingAffinityDatasetBuilder
            but with binary y (0/1).
        """
        from .bioactivity_fetcher import ChEMBLBioactivityFetcher
        from .featurizers import CRAFTTrainingFeaturizer

        fetcher = ChEMBLBioactivityFetcher()
        featurizer = CRAFTTrainingFeaturizer()

        # Pre-generate CRAFT vectors
        craft_cache = {}
        for t in targets:
            ident = t.get('uniprot_id') or t.get('chembl_id')
            craft_cache[t['chembl_id']] = _require_craft_vector(
                featurizer, ident, t.get('gene', ident))

        # Fetch and binarize per target
        target_actives = {}     # chembl_id → set of active SMILES
        target_inactives = {}   # chembl_id → set of inactive SMILES
        all_compound_data = {}  # smiles → {chembl_id: label, ...}

        for t in targets:
            chembl_id = t['chembl_id']
            if chembl_id not in craft_cache:
                continue

            records = fetcher.fetch_bioactivities(
                chembl_id, max_value_nm=inactive_threshold_nm * 2,
                limit=max_per_target * 2)

            actives = set()
            inactives = set()

            for r in records:
                if r.standard_value < active_threshold_nm:
                    actives.add(r.smiles)
                elif r.standard_value > inactive_threshold_nm:
                    inactives.add(r.smiles)

                if r.smiles not in all_compound_data:
                    all_compound_data[r.smiles] = {}
                all_compound_data[r.smiles][chembl_id] = (
                    1 if r.standard_value < active_threshold_nm else 0)

            target_actives[chembl_id] = actives
            target_inactives[chembl_id] = inactives
            logger.info("  %s: %d active, %d inactive",
                        t.get('gene', chembl_id), len(actives), len(inactives))

        # Build dataset with cross-target negatives
        all_smiles = []
        all_labels = []
        all_target_ids = []
        all_target_genes = []
        all_craft = []

        active_targets = [t for t in targets if t['chembl_id'] in target_actives]

        for t in active_targets:
            chembl_id = t['chembl_id']
            craft_vec = craft_cache[chembl_id]

            # Positives: compounds active at this target
            # Set order depends on PYTHONHASHSEED, so sort first (deterministic), then
            # shuffle with a fixed seed so the truncation below keeps a random-like
            # subset rather than the alphabetically-first SMILES (biased).
            import random
            positives = sorted(target_actives[chembl_id])
            random.Random(42).shuffle(positives)
            for smi in positives[:max_per_target]:
                all_smiles.append(smi)
                all_labels.append(1)
                all_target_ids.append(chembl_id)
                all_target_genes.append(t.get('gene', ''))
                all_craft.append(craft_vec)

            n_pos = min(len(positives), max_per_target)
            n_neg_target = n_pos  # 1:1 ratio

            # Negatives: 70% cross-target, 30% random inactive
            n_cross = int(n_neg_target * 0.7)
            n_inactive = n_neg_target - n_cross

            # Cross-target negatives: active at OTHER targets, not at this one
            cross_negatives = []
            for other_t in active_targets:
                if other_t['chembl_id'] == chembl_id:
                    continue
                other_actives = target_actives[other_t['chembl_id']]
                for smi in sorted(other_actives):
                    if smi not in target_actives[chembl_id]:
                        cross_negatives.append(smi)

            import random
            rng = random.Random(42)
            rng.shuffle(cross_negatives)
            for smi in cross_negatives[:n_cross]:
                all_smiles.append(smi)
                all_labels.append(0)
                all_target_ids.append(chembl_id)
                all_target_genes.append(t.get('gene', ''))
                all_craft.append(craft_vec)

            # Random inactive negatives
            inactive_list = sorted(target_inactives.get(chembl_id, set()))
            rng.shuffle(inactive_list)
            for smi in inactive_list[:n_inactive]:
                all_smiles.append(smi)
                all_labels.append(0)
                all_target_ids.append(chembl_id)
                all_target_genes.append(t.get('gene', ''))
                all_craft.append(craft_vec)

        if not all_smiles:
            raise ValueError("No polypharmacology data constructed")

        # Featurize
        X_ecfp = featurizer.featurize_ecfp(all_smiles)
        craft_matrix = np.array(all_craft, dtype=np.float32)
        X_craft_augmented = featurizer.fuse_ecfp_craft(X_ecfp, craft_matrix)
        y = np.array(all_labels, dtype=np.float32)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        logger.info("Polypharmacology dataset: %d samples (%d pos, %d neg)",
                     len(y), n_pos, n_neg)

        result = {
            'X_ecfp': X_ecfp,
            'X_craft_augmented': X_craft_augmented,
            'y': y,
            'smiles': all_smiles,
            'target_ids': all_target_ids,
            'target_genes': all_target_genes,
            'metadata': {
                'n_samples': len(y),
                'n_positive': n_pos,
                'n_negative': n_neg,
                'n_targets': len(active_targets),
                'active_threshold_nm': active_threshold_nm,
                'inactive_threshold_nm': inactive_threshold_nm,
            },
        }

        if include_baselines:
            # Reuse BindingAffinityDatasetBuilder's baseline builder
            ba_builder = BindingAffinityDatasetBuilder()
            result.update(ba_builder._build_baseline_matrices(
                X_ecfp, all_target_ids, targets))

        return result


class LOFODatasetBuilder:
    """
    Build Leave-One-Family-Out datasets for generalization experiment (Task 6).

    Partitions targets by protein family, holds out one family for testing,
    and builds train/test datasets with all feature representations.
    """

    def build_train(self, targets: List[Dict], max_per_target: int = 2000,
                    max_value_nm: float = 50000) -> Dict:
        """Build training dataset from given targets."""
        ba_builder = BindingAffinityDatasetBuilder()
        return ba_builder.build(targets, max_per_target=max_per_target,
                                max_value_nm=max_value_nm,
                                include_baselines=True)

    def build_test(self, targets: List[Dict], max_per_target: int = 2000,
                   max_value_nm: float = 50000) -> Dict:
        """Build test dataset from held-out family targets."""
        ba_builder = BindingAffinityDatasetBuilder()
        return ba_builder.build(targets, max_per_target=max_per_target,
                                max_value_nm=max_value_nm,
                                include_baselines=True)

    def build_onehot_matrix(self, target_ids: List[str],
                             all_train_targets: List[str]) -> np.ndarray:
        """
        Build one-hot target encoding matrix.

        For held-out family targets: all columns are zero (unseen targets).
        This is the key insight - one-hot fails for unseen families.

        Args:
            target_ids: Per-sample target identifiers
            all_train_targets: List of all training target IDs (defines column order)

        Returns:
            np.ndarray of shape (n_samples, n_train_targets)
        """
        target_to_idx = {t: i for i, t in enumerate(sorted(set(all_train_targets)))}
        n_targets = len(target_to_idx)
        n = len(target_ids)
        onehot = np.zeros((n, n_targets), dtype=np.float32)
        for i, tid in enumerate(target_ids):
            if tid in target_to_idx:
                onehot[i, target_to_idx[tid]] = 1.0
            # Held-out targets: row stays all zeros
        return onehot
