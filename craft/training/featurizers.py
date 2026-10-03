"""
CRAFT Training Featurizer - ECFP4 + CRAFT vector generation for validation.

Self-contained: does NOT import from poly_x. Replicates ECFP4 generation
from craft_fusioner.py using the RDKit new API (rdFingerprintGenerator).
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CRAFTTrainingFeaturizer:
    """
    Batch featurizer for CRAFT validation tasks.

    Generates ECFP4 fingerprints for compounds and CRAFT vectors for targets,
    with optional fusion into 2304-bit drug-in-context representations.
    """

    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius
        self._craft_cache: Dict[str, np.ndarray] = {}

        # Initialize RDKit fingerprint generator
        from rdkit.Chem import rdFingerprintGenerator
        self._fp_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits)

    def featurize_ecfp(self, smiles_list: List[str]) -> np.ndarray:
        """
        Generate ECFP4 fingerprint matrix.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            np.ndarray of shape (n_samples, n_bits) with float32 values.
            Invalid molecules get zero vectors.
        """
        from rdkit import Chem

        n = len(smiles_list)
        fps = np.zeros((n, self.n_bits), dtype=np.float32)
        n_failed = 0

        for i, smi in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = self._fp_gen.GetFingerprintAsNumPy(mol)
                    fps[i] = fp.astype(np.float32)
                else:
                    n_failed += 1
            except Exception:
                n_failed += 1

        if n_failed > 0:
            logger.warning("Failed to featurize %d/%d molecules (%.1f%%)",
                           n_failed, n, 100 * n_failed / n if n else 0)

        return fps

    def generate_craft_vector(self, identifier: str) -> Optional[np.ndarray]:
        """
        Get CRAFT 256-bit vector for a target identifier.

        Checks memory cache → CRAFTDatabase → CRAFTGenerator (API call).

        Args:
            identifier: UniProt ID, ChEMBL ID, or gene name

        Returns:
            np.ndarray of shape (256,) with float32 values, or None on failure
        """
        if identifier in self._craft_cache:
            return self._craft_cache[identifier]

        # Try pre-computed database first
        try:
            from craft.services.craft_database import CRAFTDatabase
            db = CRAFTDatabase()
            entry = db.lookup(identifier)
            if entry and entry.get('vector'):
                vec = np.array(entry['vector'], dtype=np.float32)
                self._craft_cache[identifier] = vec
                return vec
        except Exception as e:
            logger.debug("Database lookup failed for %s: %s", identifier, e)

        # Generate via API
        try:
            from craft.services.craft_generator import CRAFTGenerator
            gen = CRAFTGenerator()
            fp = gen.generate(identifier, mode='binary')
            if fp.success and fp.vector is not None:
                vec = fp.vector.astype(np.float32)
                self._craft_cache[identifier] = vec
                logger.info("Generated CRAFT for %s: %d ON bits",
                            identifier, int(vec.sum()))
                return vec
            else:
                logger.warning("CRAFT generation failed for %s: %s",
                               identifier, fp.error)
                return None
        except Exception as e:
            logger.error("CRAFT generation error for %s: %s", identifier, e)
            return None

    def batch_generate_craft(self, identifiers: List[str]) -> Dict[str, np.ndarray]:
        """
        Generate CRAFT vectors for multiple targets.

        Args:
            identifiers: List of target identifiers

        Returns:
            Dict mapping identifier to 256-dim vector (skips failures)
        """
        results = {}
        for ident in identifiers:
            vec = self.generate_craft_vector(ident)
            if vec is not None:
                results[ident] = vec
            else:
                logger.warning("Skipping target %s (CRAFT generation failed)", ident)
        return results

    @staticmethod
    def fuse_ecfp_craft(ecfp_matrix: np.ndarray,
                        craft_vectors: np.ndarray) -> np.ndarray:
        """
        Concatenate ECFP and CRAFT matrices.

        Args:
            ecfp_matrix: Shape (n, 2048) - compound fingerprints
            craft_vectors: Shape (n, 256) - target CRAFT vectors
                          (each row is the CRAFT vector of that sample's target)

        Returns:
            np.ndarray of shape (n, 2304) - fused drug-in-context representation
        """
        assert ecfp_matrix.shape[0] == craft_vectors.shape[0], \
            f"Row mismatch: ECFP {ecfp_matrix.shape[0]} vs CRAFT {craft_vectors.shape[0]}"
        return np.hstack([ecfp_matrix, craft_vectors]).astype(np.float32)

    def get_valid_mask(self, fps: np.ndarray) -> np.ndarray:
        """Return boolean mask for non-zero fingerprint rows."""
        return fps.sum(axis=1) > 0
