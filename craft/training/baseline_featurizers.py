"""
Baseline featurizers for CRAFT validation - competitive controls.

Three ablation baselines to prove CRAFT's improvement comes from
biological content, not dimensionality expansion or generic features:

1. RandomBitFeaturizer: 256 random bits per target (dimensionality control)
2. ESM2Featurizer: ESM-2 650M protein sequence embeddings (1280-D)
3. MetadataFeaturizer: Naive database features (~30-D from UniProt/ChEMBL)
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# Baseline 1: Random 256-bit vectors (dimensionality ablation)
# ───────────────────────────────────────────────────────────────────────────

class RandomBitFeaturizer:
    """
    Generate deterministic random 256-bit vectors per target.

    ECFP4 (2048) + random (256) = 2304 features - same dimensionality as CRAFT.
    If CRAFT > random, the improvement is from biological content, not extra features.
    """

    FEATURE_DIM = 256

    def generate(self, identifier: str) -> np.ndarray:
        """
        Generate a deterministic 256-bit random binary vector.

        Uses SHA-256 hash of identifier as seed for reproducibility.
        Same identifier always produces the same vector.
        """
        # Deterministic seed from identifier
        hash_bytes = hashlib.sha256(identifier.encode('utf-8')).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder='big')
        rng = np.random.RandomState(seed)

        # Binary vector with ~50% density (like typical CRAFT vectors)
        return rng.randint(0, 2, size=self.FEATURE_DIM).astype(np.float32)

    def batch_generate(self, identifiers: List[str]) -> Dict[str, np.ndarray]:
        """Generate random vectors for multiple identifiers."""
        return {ident: self.generate(ident) for ident in identifiers}


# ───────────────────────────────────────────────────────────────────────────
# Baseline 2: ESM-2 protein sequence embeddings
# ───────────────────────────────────────────────────────────────────────────

class ESM2Featurizer:
    """
    ESM-2 650M (esm2_t33_650M_UR50D) mean-pooled embeddings via HuggingFace transformers.

    ECFP4 (2048) + ESM-2 (1280) = 3328 features.
    Tests whether learned sequence representations outperform curated biology.

    Sequences fetched from UniProt, embeddings computed with transformers.
    Both are disk-cached for efficiency.

    Requires: transformers >= 4.20, torch
    """

    MODEL_NAME = "esm2_t33_650M_UR50D"
    HF_MODEL_ID = "facebook/esm2_t33_650M_UR50D"
    EMBEDDING_DIM = 1280
    MAX_TOKENS = 1022  # ESM-2 max sequence length (excl. special tokens)

    def __init__(self, cache_dir: Optional[str] = None):
        self._model = None
        self._tokenizer = None

        if cache_dir is None:
            from craft import config
            base = config.cache_dir('esm2')
        else:
            base = Path(cache_dir)
        self._cache_dir = base
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # Refuse to load the 2.5 GB ESM-2 model if system RAM is above this threshold.
    # PyTorch's C extension crashes the process (SIGSEGV / Windows os error 1455)
    # when the paging file is exhausted - that cannot be caught by Python try/except,
    # so the check must happen *before* calling AutoModel.from_pretrained.
    # Conservative threshold (75%): ESM-2 model (2.5 GB) + ADMET matrices (1.2 GB)
    # together require ~3.7 GB of free RAM. On a 31 GB system, safe headroom
    # requires system RAM ≤ 63%. Since validation processes themselves use ~1 GB,
    # 75% is the practical upper bound to avoid swap-thrashing.
    _MEM_LOAD_THRESHOLD = 0.75  # 75 % system RAM used → skip load

    def _load_model(self):
        """Lazy-load ESM-2 model via transformers (first call downloads ~2.5 GB)."""
        if self._model is not None:
            return

        import psutil
        mem = psutil.virtual_memory()
        if mem.percent / 100.0 >= self._MEM_LOAD_THRESHOLD:
            raise MemoryError(
                f"System RAM at {mem.percent:.1f}% - refusing to load ESM-2 "
                f"({self.HF_MODEL_ID}) to avoid paging-file crash. "
                f"Free memory: {mem.available / (1024**3):.1f} GB"
            )

        import torch
        from transformers import AutoTokenizer, AutoModel

        logger.info("Loading ESM-2 model via transformers (%s)...", self.HF_MODEL_ID)
        self._tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL_ID)
        self._model = AutoModel.from_pretrained(self.HF_MODEL_ID)
        self._model.eval()
        logger.info("ESM-2 loaded successfully (CPU mode)")

    def fetch_sequence(self, uniprot_id: str) -> Optional[str]:
        """
        Fetch protein sequence from UniProt FASTA endpoint.

        Disk-cached to avoid repeated API calls.
        """
        cache_path = self._cache_dir / f"seq_{uniprot_id}.txt"

        # Check cache
        if cache_path.exists():
            seq = cache_path.read_text().strip()
            if seq:
                return seq

        # Fetch from UniProt
        try:
            import requests
            url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                lines = resp.text.strip().split('\n')
                seq = ''.join(line.strip() for line in lines
                              if not line.startswith('>'))
                if seq:
                    cache_path.write_text(seq)
                    return seq
            else:
                logger.warning("UniProt FASTA fetch failed for %s: HTTP %d",
                               uniprot_id, resp.status_code)
        except Exception as e:
            logger.error("UniProt FASTA error for %s: %s", uniprot_id, e)

        return None

    def generate(self, uniprot_id: str) -> Optional[np.ndarray]:
        """
        Generate ESM-2 embedding for a protein target.

        Returns mean-pooled 1280-D float vector (excluding BOS/EOS tokens).
        Disk-cached as .npy file.
        """
        cache_path = self._cache_dir / f"emb_{uniprot_id}.npy"

        # Check embedding cache
        if cache_path.exists():
            return np.load(str(cache_path))

        # Get sequence
        seq = self.fetch_sequence(uniprot_id)
        if not seq:
            return None

        # Truncate to max tokens
        if len(seq) > self.MAX_TOKENS:
            logger.info("Truncating %s sequence from %d to %d residues",
                        uniprot_id, len(seq), self.MAX_TOKENS)
            seq = seq[:self.MAX_TOKENS]

        # Load model and compute embedding
        self._load_model()

        import torch

        try:
            inputs = self._tokenizer(
                seq,
                return_tensors='pt',
                truncation=True,
                max_length=self.MAX_TOKENS + 2,  # +2 for [CLS]/[EOS] special tokens
            )

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Mean pool over sequence positions (exclude [CLS] at 0 and [EOS] at -1)
            # outputs.last_hidden_state shape: (1, seq_len+2, 1280)
            token_repr = outputs.last_hidden_state[0, 1:-1, :]  # strip special tokens
            embedding = token_repr.mean(dim=0).numpy().astype(np.float32)

            # Cache
            np.save(str(cache_path), embedding)
            logger.info("ESM-2 embedding for %s: shape %s", uniprot_id, embedding.shape)
            return embedding

        except Exception as e:
            logger.error("ESM-2 embedding error for %s: %s", uniprot_id, e)
            return None

    def batch_generate(self, uniprot_ids: List[str]) -> Dict[str, np.ndarray]:
        """Generate ESM-2 embeddings for multiple targets."""
        results = {}
        for uid in uniprot_ids:
            emb = self.generate(uid)
            if emb is not None:
                results[uid] = emb
            else:
                logger.warning("Skipping ESM-2 for %s (generation failed)", uid)
        return results


# ───────────────────────────────────────────────────────────────────────────
# Baseline 3: Naive metadata features
# ───────────────────────────────────────────────────────────────────────────

# Family categories matching CRAFT schema A4 sub-block
FAMILY_CATEGORIES = [
    'gpcr', 'kinase', 'nuclear_receptor', 'ion_channel', 'enzyme',
    'transporter', 'protease', 'phosphatase', 'transferase', 'oxidoreductase',
    'ligase', 'isomerase', 'lyase', 'hydrolase', 'other',
]

# Subcellular compartments
COMPARTMENTS = [
    'plasma_membrane', 'nucleus', 'cytoplasm', 'mitochondria',
    'endoplasmic_reticulum', 'golgi', 'lysosome', 'peroxisome',
    'extracellular', 'other',
]


# Vocabulary actually used by ChEMBL classes and UniProt locations
_FAMILY_TERMS = {
    'gpcr': ('g protein-coupled', 'gpcr'),
    'nuclear_receptor': ('nuclear receptor',),
    'ion_channel': ('ion channel',),
}
_COMPARTMENT_TERMS = {
    'plasma_membrane': ('cell membrane', 'plasma membrane'),
    'nucleus': ('nucleus',),
    'cytoplasm': ('cytoplasm', 'cytosol'),
    'mitochondria': ('mitochondri',),
    'endoplasmic_reticulum': ('endoplasmic reticulum',),
    'golgi': ('golgi',),
    'lysosome': ('lysosome',),
    'peroxisome': ('peroxisome',),
    'extracellular': ('secreted', 'extracellular'),
    'other': (),
}


class MetadataFeaturizer:
    """
    Simple features from UniProt/ChEMBL metadata (~30 dimensions).

    Features:
    - Family one-hot (15 categories)
    - Subcellular location one-hot (10 compartments)
    - Sequence length (normalized)
    - Molecular weight (normalized, derived from sequence)
    - Number of transmembrane helices (normalized)

    Tests whether CRAFT's structured 256-bit encoding outperforms
    naive database features extracted from the same data sources.
    """

    FEATURE_DIM = len(FAMILY_CATEGORIES) + len(COMPARTMENTS) + 3  # 28

    _MAX_CACHE_SIZE = 256

    def __init__(self):
        self._cache: Dict[str, np.ndarray] = {}

    def generate(self, identifier: str) -> Optional[np.ndarray]:
        """
        Generate metadata feature vector for a target.

        Uses CRAFT's existing API clients for data retrieval.
        """
        if identifier in self._cache:
            return self._cache[identifier]

        try:
            features = np.zeros(self.FEATURE_DIM, dtype=np.float32)

            # Use CRAFT's existing clients for data retrieval
            from craft.services.chembl_client import ChEMBLClient
            from craft.services.uniprot_client import UniProtClient

            chembl = ChEMBLClient()
            uniprot = UniProtClient()

            # Resolve identifier - returns ChEMBLTargetData dataclass
            target_info = chembl.resolve_target(identifier)
            if not target_info or not getattr(target_info, 'uniprot_id', ''):
                if len(self._cache) >= self._MAX_CACHE_SIZE:
                    del self._cache[next(iter(self._cache))]
                self._cache[identifier] = features
                return features

            uniprot_id = target_info.uniprot_id

            # Family one-hot
            family_str = self._infer_family(target_info)
            for i, fam in enumerate(FAMILY_CATEGORIES):
                if fam in family_str:
                    features[i] = 1.0
                    break

            # UniProt data
            try:
                up_data = uniprot.fetch(uniprot_id)
                if up_data and up_data.success:
                    # Subcellular location one-hot
                    offset = len(FAMILY_CATEGORIES)
                    locations = getattr(up_data, 'subcellular_locations', []) or []
                    loc_str = ' '.join(str(l).lower() for l in locations)
                    for i, comp in enumerate(COMPARTMENTS):
                        if any(k in loc_str for k in _COMPARTMENT_TERMS.get(comp, ())):
                            features[offset + i] = 1.0

                    # Continuous features (offset after compartments)
                    cont_offset = offset + len(COMPARTMENTS)

                    # Sequence length (log-normalized, typical range 100-5000)
                    seq_len = getattr(up_data, 'sequence_length', 0) or 0
                    if seq_len > 0:
                        features[cont_offset] = min(np.log10(seq_len) / 4.0, 1.0)

                    # MW (derived from sequence length, ~110 Da per residue)
                    if seq_len > 0:
                        mw = seq_len * 110.0
                        features[cont_offset + 1] = min(np.log10(mw) / 6.0, 1.0)

                    # TM helices count (normalized to 0-1)
                    n_tm = getattr(up_data, 'transmembrane_count', 0) or 0
                    features[cont_offset + 2] = min(n_tm / 7.0, 1.0)

            except Exception as e:
                # v0.2 logged this at debug level, which hid an AttributeError
                logger.warning("UniProt data fetch failed for %s: %s", uniprot_id, e)

            # Evict oldest entries if cache exceeds bound
            if len(self._cache) >= self._MAX_CACHE_SIZE:
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[identifier] = features
            return features

        except Exception as e:
            logger.error("Metadata featurization error for %s: %s", identifier, e)
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

    def _infer_family(self, target_info) -> str:
        """Infer family string from ChEMBLTargetData dataclass."""
        # Check L2 then L1 classification fields (ChEMBL vocabulary)
        for attr in ('target_class_l2', 'target_class_l1'):
            cls_str = getattr(target_info, attr, '').lower()
            if cls_str:
                for fam in FAMILY_CATEGORIES:
                    if any(t in cls_str for t in _FAMILY_TERMS.get(fam, (fam,))):
                        return fam

        # Fallback: check target type
        target_type = getattr(target_info, 'target_type', '').lower()
        for fam in FAMILY_CATEGORIES:
            if fam in target_type:
                return fam

        return 'other'

    def batch_generate(self, identifiers: List[str]) -> Dict[str, np.ndarray]:
        """Generate metadata features for multiple targets."""
        results = {}
        for ident in identifiers:
            vec = self.generate(ident)
            if vec is not None:
                results[ident] = vec
        return results
