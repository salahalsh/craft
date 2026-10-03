"""
CRAFT Similarity - Tanimoto-based similarity between CRAFT vectors.

Supports:
  - Binary Tanimoto: |A & B| / |A | B|
  - Fuzzy Tanimoto: sum(min(a,b)) / sum(max(a,b))
  - Per-module similarity breakdown
  - Top-N similar target search (against pre-computed database)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

from .craft_schema import MODULE_RANGES

logger = logging.getLogger(__name__)

# Module-level cache for the pre-computed vector database (loaded once, reused across calls).
# Stored as a list of dicts; vectors are kept as Python lists. The cache is bounded
# to the database size (fixed at build time) and invalidated only on process restart.
_vector_cache = None  # List[Dict] from CRAFTDatabase.get_all_vectors()


@dataclass
class SimilarTarget:
    """A similar target found in the database."""
    target_id: str = ''
    gene_name: str = ''
    target_type: str = ''
    overall_similarity: float = 0.0
    module_similarities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class SimilarityResult:
    """Result of a similarity search."""
    query_id: str = ''
    similar_targets: List[SimilarTarget] = field(default_factory=list)
    total_searched: int = 0
    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'error': self.error,
            'query_id': self.query_id,
            'similar_targets': [t.to_dict() for t in self.similar_targets],
            'total_searched': self.total_searched,
        }


class CRAFTSimilarity:
    """Compute similarity between CRAFT fingerprint vectors."""

    def tanimoto(self, vec_a: np.ndarray, vec_b: np.ndarray,
                 mode: str = 'binary') -> float:
        """
        Compute Tanimoto similarity between two CRAFT vectors.

        Binary mode: |A & B| / |A | B|
        Fuzzy mode: sum(min(a,b)) / sum(max(a,b))
        """
        if vec_a is None or vec_b is None:
            return 0.0

        if mode == 'fuzzy':
            min_sum = np.minimum(vec_a, vec_b).sum()
            max_sum = np.maximum(vec_a, vec_b).sum()
            return float(min_sum / max_sum) if max_sum > 0 else 0.0
        else:
            a = vec_a.astype(bool)
            b = vec_b.astype(bool)
            intersection = np.logical_and(a, b).sum()
            union = np.logical_or(a, b).sum()
            # When both vectors are all-zero (e.g. missing IUPHAR data for a module),
            # union=0 → return 0.0 (no evidence of similarity, not identical).
            return float(intersection / union) if union > 0 else 0.0

    def module_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray,
                          mode: str = 'binary') -> Dict[str, float]:
        """Compute per-module Tanimoto similarity."""
        result = {}
        for mod_key, (start, end) in MODULE_RANGES.items():
            mod_a = vec_a[start:end + 1]
            mod_b = vec_b[start:end + 1]
            result[f'module_{mod_key.lower()}'] = self.tanimoto(mod_a, mod_b, mode)
        return result

    def compare_pair(self, fp_a, fp_b) -> Dict:
        """
        Compare two CRAFTFingerprint objects.

        Returns dict with overall + per-module similarities.
        """
        mode = fp_a.mode if fp_a.mode == fp_b.mode else 'binary'
        overall = self.tanimoto(fp_a.vector, fp_b.vector, mode)
        modules = self.module_similarity(fp_a.vector, fp_b.vector, mode)

        return {
            'overall': overall,
            **modules,
            'target_a': fp_a.target_id,
            'target_b': fp_b.target_id,
            'mode': mode,
        }

    def find_similar(self, fingerprint, top_n: int = 10) -> SimilarityResult:
        """
        Find most similar targets in the pre-computed database.

        Args:
            fingerprint: CRAFTFingerprint query
            top_n: Number of results to return

        Returns:
            SimilarityResult with ranked similar targets
        """
        result = SimilarityResult(query_id=fingerprint.target_id)

        try:
            from .craft_database import CRAFTDatabase
            db = CRAFTDatabase()

            if not db.is_available():
                result.error = "Pre-computed database not built yet"
                return result

            global _vector_cache
            if _vector_cache is None:
                _vector_cache = db.get_all_vectors()
            all_targets = _vector_cache
            result.total_searched = len(all_targets)

            if not all_targets:
                return result

            mode = fingerprint.mode
            query_vec = fingerprint.vector

            scored = []
            for entry in all_targets:
                target_id = entry['target_id']
                if target_id == fingerprint.target_id:
                    continue

                ref_vec = np.array(entry['vector'], dtype=query_vec.dtype)
                sim = self.tanimoto(query_vec, ref_vec, mode)
                mod_sim = self.module_similarity(query_vec, ref_vec, mode)

                scored.append(SimilarTarget(
                    target_id=target_id,
                    gene_name=entry.get('gene_name', ''),
                    target_type=entry.get('target_type', ''),
                    overall_similarity=round(sim, 4),
                    module_similarities={k: round(v, 4) for k, v in mod_sim.items()},
                ))

            scored.sort(key=lambda x: x.overall_similarity, reverse=True)
            result.similar_targets = scored[:top_n]

        except Exception as e:
            logger.warning("Similarity search failed: %s", e)
            result.error = f"Similarity search failed: {e}"

        return result
