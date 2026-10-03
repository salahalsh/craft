"""
PDBe REST API Client - Binding site data for CRAFT Module C Tier 1.

Retrieves real binding site residue composition from PDB structures via the
PDBe REST API (https://www.ebi.ac.uk/pdbe/api). No new dependencies required.

Fallback chain preserved: PDB (Tier 1) → class-based (Tier 2) → unknown (Tier 3).
Follows chembl_client.py / uniprot_client.py pattern: rate-limited, disk-cached.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


# Amino acid classification by physicochemical properties
HYDROPHOBIC = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO'}
POLAR = {'SER', 'THR', 'ASN', 'GLN', 'CYS'}
CHARGED_POS = {'LYS', 'ARG', 'HIS'}
CHARGED_NEG = {'ASP', 'GLU'}
AROMATIC = {'PHE', 'TRP', 'TYR'}
SPECIAL = {'GLY'}  # GLY: smallest AA, neither hydrophobic nor polar
ALL_STANDARD = HYDROPHOBIC | POLAR | CHARGED_POS | CHARGED_NEG | AROMATIC | SPECIAL

# Common non-drug ligands to exclude when selecting "best" structure
# (sugars, cofactors, crystallization additives, buffer molecules, lipids)
NON_DRUG_LIGANDS = {
    'HOH', 'GOL', 'EDO', 'DMS', 'SO4', 'PO4', 'ACT', 'MES', 'TRS',  # Buffers
    'IPA', 'ACY', 'FLC', 'CIT', 'BME', 'DTT', 'ACM', 'FMT',          # Small additives
    'BU1', 'BU2', 'BU3', 'BOG', 'LMT', 'C8E', 'LDA', 'UNL',          # Crystallization aids
    'NAG', 'BMA', 'MAN', 'FUC', 'GAL', 'GLC', 'SIA', 'BGC',          # Sugars
    'NAD', 'FAD', 'FMN', 'SAM', 'SAH', 'COA', 'HEM', 'HEC',          # Cofactors
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'UTP', 'CTP',                  # Nucleotides
    'ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'NI', 'CO',                  # Metals
    'CL', 'BR', 'IOD', 'SCN', 'K', 'NA', 'I', 'F',                    # Ions/elements
    'PEG', 'PGE', 'P6G', '1PE', 'MPD', 'EPE', '12P', 'P33', 'P4G',  # PEG fragments
    'CLR', 'PLM', 'OLA', 'OLC', 'SPH', 'CER', 'LNR', 'CDL',          # Lipids
}


@dataclass
class PDBBindingSiteData:
    """Binding site characterization from PDB structure."""
    pdb_id: str = ''
    resolution: float = 0.0
    method: str = ''               # 'X-ray diffraction', 'Electron Microscopy', 'NMR'
    has_ligand: bool = False
    ligand_id: str = ''
    n_residues: int = 0            # Binding site residue count
    n_hydrophobic: int = 0
    n_polar: int = 0
    n_charged_pos: int = 0
    n_charged_neg: int = 0
    n_aromatic: int = 0
    frac_hydrophobic: float = 0.0
    frac_polar: float = 0.0
    frac_charged: float = 0.0
    frac_aromatic: float = 0.0
    pocket_size_category: str = ''   # 'small', 'medium', 'large', 'very_large'
    has_cysteine_in_site: bool = False
    has_metal_binding: bool = False
    has_disulfide_near_site: bool = False
    residue_names: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()
                if k != 'residue_names'}


class PDBEClient:
    """
    PDBe REST API client for binding site characterization.

    Retrieves the best PDB structure for a UniProt ID, extracts binding site
    residues, and computes pocket physicochemical properties for CRAFT Module C.
    """

    BASE_URL = 'https://www.ebi.ac.uk/pdbe/api'

    def __init__(self, cache_dir: Optional[str] = None,
                 cache_ttl_hours: Optional[int] = None):
        if cache_dir is None:
            from craft import config
            base = config.cache_dir('pdbe')
        else:
            base = Path(cache_dir)
        self.cache_dir = base
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from craft import config
        self.cache_ttl = (config.cache_ttl_hours() if cache_ttl_hours is None
                          else cache_ttl_hours) * 3600

        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'CRAFT/1.0 (InsilicoSigma)',
        })
        self._last_request_time = 0

    def close(self):
        """Close the underlying HTTP session to release connections."""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __del__(self):
        self.close()

    # ── Public API ────────────────────────────────────────────────────────

    def get_best_structure(self, uniprot_id: str) -> Optional[str]:
        """
        Find the best drug-bound PDB structure for a UniProt accession.

        Two-pass strategy (PDBe best_structures ranks by coverage, so drug-bound
        kinase-domain structures can appear at rank 200+ behind full-length cryo-EM):

        Pass 1: Extract unique X-ray structures, sort by resolution, check top 15
                for drug-like ligands. Select best drug-bound X-ray.
        Pass 2: If no drug-bound X-ray found, fall back to highest-resolution
                X-ray (for class-based pocket fallback in Module C).
        """
        cache_key = f"best_pdb_{uniprot_id}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached.get('pdb_id')

        # Use best_structures endpoint (pre-ranked by PDBe)
        url = f"{self.BASE_URL}/mappings/best_structures/{uniprot_id}"
        data = self._request(url)
        if not data or uniprot_id not in data:
            self._set_cache(cache_key, {'pdb_id': None})
            return None

        entries = data[uniprot_id]
        if not entries:
            self._set_cache(cache_key, {'pdb_id': None})
            return None

        # Deduplicate and extract unique X-ray structures sorted by resolution
        seen_pdbs = set()
        xray_by_resolution = []
        for entry in entries:
            pdb_id = entry.get('pdb_id', '').lower()
            if pdb_id in seen_pdbs:
                continue
            seen_pdbs.add(pdb_id)

            method = entry.get('experimental_method', '')
            if 'X-ray' not in method:
                continue

            resolution = entry.get('resolution', 99.0) or 99.0
            if resolution > 4.0:  # Skip very low resolution
                continue

            xray_by_resolution.append({
                'pdb_id': pdb_id,
                'resolution': resolution,
                'method': method,
            })

        # Sort by resolution (ascending - lower is better for pocket analysis)
        xray_by_resolution.sort(key=lambda x: x['resolution'])

        # Pass 1: Check top 15 X-ray structures (by resolution) for drug ligands
        drug_bound = None
        for candidate in xray_by_resolution[:15]:
            if self._has_drug_ligand(candidate['pdb_id']):
                drug_bound = candidate
                break

        if drug_bound:
            best = drug_bound
            best['has_ligand'] = True
        elif xray_by_resolution:
            # Pass 2: No drug-bound found - use best-resolution X-ray anyway
            best = xray_by_resolution[0]
            best['has_ligand'] = False
        else:
            self._set_cache(cache_key, {'pdb_id': None})
            return None

        self._set_cache(cache_key, {'pdb_id': best['pdb_id'],
                                     'resolution': best['resolution'],
                                     'method': best['method']})

        logger.info("Best PDB for %s: %s (%.1fÅ, %s, drug_ligand=%s)",
                     uniprot_id, best['pdb_id'], best['resolution'],
                     best['method'], best.get('has_ligand', False))
        return best['pdb_id']

    def get_binding_site(self, pdb_id: str) -> Optional[PDBBindingSiteData]:
        """
        Extract binding site residue composition from a PDB entry.

        Strategy:
        1. Find bound molecules via graph-api bound_molecules endpoint
        2. Get ligand-protein interactions via bound_molecule_interactions
        3. Extract interacting protein residues and classify by property
        """
        cache_key = f"site_{pdb_id}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            if cached.get('success') is False:
                return None
            return self._dict_to_site_data(cached)

        # Step 1: Get bound molecules
        bm_url = f"https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_molecules/{pdb_id}"
        bm_data = self._request(bm_url)

        bound_molecules = []
        if bm_data and pdb_id in bm_data:
            bound_molecules = bm_data[pdb_id]

        if not bound_molecules:
            self._set_cache(cache_key, {'success': False})
            return None

        # Step 2: Collect all drug-like bound molecules
        drug_bms = []
        for bm in bound_molecules:
            ligands = bm.get('composition', {}).get('ligands', [])
            for lig in ligands:
                comp_id = lig.get('chem_comp_id', '')
                if comp_id and comp_id not in NON_DRUG_LIGANDS and len(comp_id) >= 2:
                    drug_bms.append((bm.get('bm_id', 'bm1'), comp_id))
                    break  # One drug ligand per bound molecule is enough

        if not drug_bms:
            # No drug-like ligand found - use first bound molecule anyway
            bm_id_fallback = bound_molecules[0].get('bm_id', 'bm1')
            ligand_info = bound_molecules[0].get('composition', {}).get('ligands', [])
            lig_fallback = ligand_info[0].get('chem_comp_id', '') if ligand_info else ''
            drug_bms = [(bm_id_fallback, lig_fallback)]

        # Step 3: For each drug BM, get interactions and pick the one with
        # the most protein residue contacts (actual drug > small additive)
        best_residues: Set[tuple] = set()
        best_ligand = ''
        for bm_id, lig_id in drug_bms[:3]:  # Check up to 3 candidates
            inter_url = (f"https://www.ebi.ac.uk/pdbe/graph-api/pdb/"
                         f"bound_molecule_interactions/{pdb_id}/{bm_id}")
            inter_data = self._request(inter_url)

            if not inter_data or pdb_id not in inter_data:
                continue

            interactions = inter_data[pdb_id]
            if not interactions:
                continue

            residues: Set[tuple] = set()
            for entry in interactions:
                for inter in entry.get('interactions', []):
                    end = inter.get('end', {})
                    comp = end.get('chem_comp_id', '')
                    resnum = end.get('author_residue_number', 0)
                    if comp in ALL_STANDARD:
                        residues.add((comp, resnum))

            if len(residues) > len(best_residues):
                best_residues = residues
                best_ligand = lig_id

        site_residues_set = best_residues

        if not site_residues_set:
            self._set_cache(cache_key, {'success': False})
            return None

        residue_names = [r[0] for r in sorted(site_residues_set, key=lambda x: x[1])]

        # Compute properties
        site_data = self._compute_site_properties(pdb_id, residue_names, best_ligand)

        self._set_cache(cache_key, site_data.to_dict())
        return site_data

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_entry_summary(self, pdb_id: str) -> Optional[Dict]:
        """Get PDB entry summary (resolution, method)."""
        cache_key = f"summary_{pdb_id}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        url = f"{self.BASE_URL}/pdb/entry/summary/{pdb_id}"
        data = self._request(url)
        if data and pdb_id.lower() in data:
            summary = data[pdb_id.lower()][0] if data[pdb_id.lower()] else {}
            self._set_cache(cache_key, summary)
            return summary
        return None

    def _has_drug_ligand(self, pdb_id: str) -> bool:
        """Check if PDB entry has a drug-like bound molecule (not sugar/buffer)."""
        cache_key = f"has_drug_{pdb_id}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached.get('has_drug', False)

        url = f"https://www.ebi.ac.uk/pdbe/graph-api/pdb/bound_molecules/{pdb_id}"
        data = self._request(url)

        has_drug = False
        if data and pdb_id in data:
            for bm in data[pdb_id]:
                ligands = bm.get('composition', {}).get('ligands', [])
                for lig in ligands:
                    comp_id = lig.get('chem_comp_id', '')
                    if comp_id and len(comp_id) >= 2 and comp_id not in NON_DRUG_LIGANDS:
                        has_drug = True
                        break
                if has_drug:
                    break

        self._set_cache(cache_key, {'has_drug': has_drug})
        return has_drug

    def _compute_site_properties(self, pdb_id: str, residue_names: List[str],
                                  ligand_id: str) -> PDBBindingSiteData:
        """Classify binding site residues and compute pocket properties."""
        n = len(residue_names)

        n_hydrophobic = sum(1 for r in residue_names if r in HYDROPHOBIC)
        n_polar = sum(1 for r in residue_names if r in POLAR)
        n_charged_pos = sum(1 for r in residue_names if r in CHARGED_POS)
        n_charged_neg = sum(1 for r in residue_names if r in CHARGED_NEG)
        n_aromatic = sum(1 for r in residue_names if r in AROMATIC)
        n_charged = n_charged_pos + n_charged_neg

        has_cys = 'CYS' in residue_names
        has_his = 'HIS' in residue_names

        frac_hydro = n_hydrophobic / n if n > 0 else 0
        frac_polar = n_polar / n if n > 0 else 0
        frac_charged = n_charged / n if n > 0 else 0
        frac_aromatic = n_aromatic / n if n > 0 else 0

        # Pocket size from residue count
        size = self._estimate_pocket_size(n)

        return PDBBindingSiteData(
            pdb_id=pdb_id,
            has_ligand=bool(ligand_id),
            ligand_id=ligand_id,
            n_residues=n,
            n_hydrophobic=n_hydrophobic,
            n_polar=n_polar,
            n_charged_pos=n_charged_pos,
            n_charged_neg=n_charged_neg,
            n_aromatic=n_aromatic,
            frac_hydrophobic=frac_hydro,
            frac_polar=frac_polar,
            frac_charged=frac_charged,
            frac_aromatic=frac_aromatic,
            pocket_size_category=size,
            has_cysteine_in_site=has_cys,
            has_metal_binding=has_his and (n_charged_neg > 0),  # HIS+ASP/GLU = metal coordination
            residue_names=residue_names,
            success=True,
        )

    def _estimate_pocket_size(self, n_residues: int) -> str:
        """Estimate pocket size category from residue count."""
        if n_residues < 10:
            return 'small'
        elif n_residues < 20:
            return 'medium'
        elif n_residues < 35:
            return 'large'
        else:
            return 'very_large'

    def _classify_residue(self, residue_name: str) -> str:
        """Classify amino acid by primary physicochemical property."""
        if residue_name in HYDROPHOBIC:
            return 'hydrophobic'
        if residue_name in POLAR:
            return 'polar'
        if residue_name in CHARGED_POS:
            return 'charged_pos'
        if residue_name in CHARGED_NEG:
            return 'charged_neg'
        if residue_name in AROMATIC:
            return 'aromatic'
        return 'other'

    def _dict_to_site_data(self, d: Dict) -> PDBBindingSiteData:
        """Reconstruct PDBBindingSiteData from cached dict."""
        result = PDBBindingSiteData()
        for k, v in d.items():
            if hasattr(result, k) and k != 'residue_names':
                setattr(result, k, v)
        return result

    # ── HTTP + Cache ──────────────────────────────────────────────────────

    def _request(self, url: str) -> Optional[Dict]:
        """Make rate-limited HTTP request with retry."""
        # Rate limit: 200ms between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

        for attempt in range(3):
            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, timeout=30)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.info("PDBe rate-limited, waiting %ds...", wait)
                    time.sleep(wait)
                else:
                    logger.warning("PDBe HTTP %d for %s", resp.status_code, url)
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    time.sleep(1)
                else:
                    logger.error("PDBe request failed after 3 attempts: %s", e)
                    return None

        return None

    def _get_cache(self, key: str) -> Optional[Dict]:
        """Read from disk cache if not expired."""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.json"
        if cache_file.exists():
            try:
                age = time.time() - cache_file.stat().st_mtime
                if age < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            except Exception:
                pass
        return None

    def _set_cache(self, key: str, data: Dict):
        """Write to disk cache."""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.debug("Cache write failed for %s: %s", key, e)
