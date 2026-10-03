"""
IUPHAR/BPS Guide to Pharmacology API Client - Target classification,
signal transduction mechanisms, and endogenous ligands.

Feeds Module B (signaling) and Module D (endogenous ligand) bits.

API: https://www.guidetopharmacology.org/services/
"""

import json
import os
import time
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

# Since 2026 the GtoPdb web services require a registered API key and answer
# every anonymous request with HTTP 401. Setting CRAFT_DISABLE_IUPHAR=1 turns
# the source off explicitly, ABOVE the cache, so that a run either uses
# IUPHAR for every target or for none. Without the switch, stale cache
# entries would give IUPHAR bits to some targets and not to others.
_IUPHAR_DISABLED = os.environ.get('CRAFT_DISABLE_IUPHAR', '').strip() in (
    '1', 'true', 'True', 'yes')
_disabled_notice_logged = False


def _iuphar_disabled_result(**kwargs) -> 'IUPHARTargetData':
    global _disabled_notice_logged
    if not _disabled_notice_logged:
        logger.warning("IUPHAR/GtoPdb is DISABLED (CRAFT_DISABLE_IUPHAR=1): "
                       "no IUPHAR data will be used for any target")
        _disabled_notice_logged = True
    return IUPHARTargetData(success=False,
                            error='IUPHAR disabled (CRAFT_DISABLE_IUPHAR=1)',
                            **kwargs)


@dataclass
class EndogenousLigandInfo:
    """Information about an endogenous ligand."""
    name: str = ''
    ligand_type: str = ''       # 'Amino acid', 'Peptide', 'Lipid', 'Amine', etc.
    ligand_id: int = 0
    approved: bool = False


@dataclass
class IUPHARTargetData:
    """Structured target data from IUPHAR/BPS."""
    target_id: int = 0
    target_name: str = ''
    target_type: str = ''               # 'GPCR', 'Ion channel', 'NHR', 'Enzyme', etc.
    family_id: int = 0
    family_name: str = ''
    target_class: str = ''              # Top-level class
    transduction_mechanisms: List[str] = field(default_factory=list)   # ['Gs', 'Gi/Go', 'Gq/G11']
    primary_pathways: List[str] = field(default_factory=list)          # ['cAMP', 'IP3/DAG']
    endogenous_ligands: List[EndogenousLigandInfo] = field(default_factory=list)
    selective_ligands_count: int = 0
    uniprot_id: str = ''
    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        d = {k: v for k, v in self.__dict__.items() if k != 'endogenous_ligands'}
        d['endogenous_ligands'] = [
            {'name': lig.name, 'ligand_type': lig.ligand_type,
             'ligand_id': lig.ligand_id, 'approved': lig.approved}
            for lig in self.endogenous_ligands
        ]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> 'IUPHARTargetData':
        ligands_raw = d.pop('endogenous_ligands', [])
        obj = cls(**d)
        obj.endogenous_ligands = [EndogenousLigandInfo(**lig) for lig in ligands_raw]
        return obj


class IUPHARClient:
    """IUPHAR Guide to Pharmacology API client."""

    BASE_URL = 'https://www.guidetopharmacology.org/services'

    # Target type mapping from IUPHAR nomenclature
    TARGET_TYPE_MAP = {
        'gpcr': 'GPCR',
        'lgic': 'Ion channel',
        'vgic': 'Ion channel',
        'other_ic': 'Ion channel',
        'nhr': 'Nuclear receptor',
        'catalytic_receptor': 'Catalytic receptor',
        'enzyme': 'Enzyme',
        'transporter': 'Transporter',
        'other_protein': 'Other protein',
    }

    # G-protein coupling types
    GPROTEIN_MAP = {
        'Gs': ['gpcr_gs_coupled'],
        'Gi/Go': ['gpcr_gi_coupled'],
        'Gq/G11': ['gpcr_gq_coupled'],
        'G12/13': ['gpcr_g12_13_coupled'],
    }

    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: Optional[int] = None):
        if cache_dir is None:
            from craft import config
            cache_dir = config.cache_dir('iuphar')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from craft import config
        self.cache_ttl = (config.cache_ttl_hours() if cache_ttl_hours is None
                          else cache_ttl_hours) * 3600
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'InsilicoSigma-CRAFT/1.0',
        })
        self._last_request_time = 0
        self._min_interval = 0.5  # Be conservative with IUPHAR

    def close(self):
        """Close the underlying HTTP session to release connections."""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __del__(self):
        self.close()

    def fetch_by_name(self, target_name: str) -> IUPHARTargetData:
        """Search IUPHAR by target name or gene symbol."""
        if _IUPHAR_DISABLED:
            return _iuphar_disabled_result(target_name=target_name)
        cached = self._get_cache(f"name_{target_name}")
        if cached:
            return IUPHARTargetData.from_dict(cached)

        data = self._api_get('/targets', params={'name': target_name})
        if data and isinstance(data, list) and len(data) > 0:
            result = self._parse_target(data[0])
            self._set_cache(f"name_{target_name}", result.to_dict())
            return result

        # Try search endpoint
        data = self._api_get('/targets', params={'name': f'%{target_name}%'})
        if data and isinstance(data, list) and len(data) > 0:
            result = self._parse_target(data[0])
            self._set_cache(f"name_{target_name}", result.to_dict())
            return result

        return IUPHARTargetData(
            target_name=target_name,
            success=False,
            error=f"Target '{target_name}' not found in IUPHAR")

    def fetch_by_uniprot(self, uniprot_id: str) -> IUPHARTargetData:
        """Search IUPHAR by UniProt accession."""
        if _IUPHAR_DISABLED:
            return _iuphar_disabled_result(uniprot_id=uniprot_id)
        cached = self._get_cache(f"uniprot_{uniprot_id}")
        if cached:
            return IUPHARTargetData.from_dict(cached)

        # Use targeted search instead of downloading the entire catalogue.
        # Try the search endpoint first, then fall back to a limited scan.
        search_data = self._api_get(f'/targets?accession={uniprot_id}')
        if search_data and isinstance(search_data, list):
            for target in search_data:
                target_detail = self._api_get(
                    f'/targets/{target.get("targetId")}')
                if target_detail:
                    db_links = target_detail.get('databaseLinks', [])
                    for link in db_links:
                        if link.get('database') == 'UniProtKB' and \
                                link.get('accession') == uniprot_id:
                            result = self._parse_target(target_detail)
                            result.uniprot_id = uniprot_id
                            self._set_cache(
                                f"uniprot_{uniprot_id}", result.to_dict())
                            return result

        # Fallback: use gene name if available
        return IUPHARTargetData(
            uniprot_id=uniprot_id,
            success=False,
            error=f"UniProt {uniprot_id} not found in IUPHAR")

    def fetch_by_gene(self, gene_name: str) -> IUPHARTargetData:
        """Search IUPHAR by gene symbol."""
        return self.fetch_by_name(gene_name)

    def _parse_target(self, data: Dict) -> IUPHARTargetData:
        """Parse IUPHAR target JSON into structured data."""
        result = IUPHARTargetData()
        result.target_id = data.get('targetId', 0)
        result.target_name = data.get('name', '')
        result.family_id = data.get('familyId', 0)
        result.family_name = data.get('familyName', '')

        # Target type from family type
        family_type = data.get('type', '').lower()
        result.target_type = self.TARGET_TYPE_MAP.get(family_type, family_type)
        result.target_class = result.target_type

        # Transduction mechanisms (for GPCRs)
        mechanisms = data.get('transductionMechanisms', [])
        for mech in mechanisms:
            coupling = mech.get('name', '')
            if coupling and coupling not in result.transduction_mechanisms:
                result.transduction_mechanisms.append(coupling)

            # Map coupling to pathway
            if 'Gs' in coupling:
                if 'cAMP' not in result.primary_pathways:
                    result.primary_pathways.append('cAMP')
            elif 'Gi' in coupling or 'Go' in coupling:
                if 'cAMP inhibition' not in result.primary_pathways:
                    result.primary_pathways.append('cAMP inhibition')
            elif 'Gq' in coupling or 'G11' in coupling:
                if 'IP3/DAG' not in result.primary_pathways:
                    result.primary_pathways.append('IP3/DAG')
                if 'calcium' not in result.primary_pathways:
                    result.primary_pathways.append('calcium')

        # Endogenous ligands
        ligands = data.get('endogenousLigands', [])
        for lig in ligands:
            info = EndogenousLigandInfo(
                name=lig.get('name', ''),
                ligand_type=lig.get('type', ''),
                ligand_id=lig.get('ligandId', 0),
                approved=lig.get('approved', False),
            )
            result.endogenous_ligands.append(info)

        return result

    def _api_get(self, endpoint: str, params: Optional[Dict] = None) -> Optional:
        """Rate-limited GET request to IUPHAR API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(3):
            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=30)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 429:
                    wait = 2 ** attempt * 5
                    logger.warning(f"IUPHAR rate limited, waiting {wait}s")
                    time.sleep(wait)
                elif resp.status_code in (401, 403):
                    # GtoPdb now requires a registered API key. Every Module B
                    # and D input from IUPHAR is lost while this persists, so
                    # it must not read like a routine per-target miss.
                    logger.error(
                        f"IUPHAR API {resp.status_code} (API key required?): "
                        f"Module B/D IUPHAR data unavailable. {url}")
                    return None
                else:
                    logger.warning(f"IUPHAR API {resp.status_code}: {url}")
                    return None

            except requests.RequestException as e:
                logger.warning(f"IUPHAR API error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt * 2)

        return None

    def _get_cache(self, key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.json"
        if cache_file.exists():
            try:
                age = time.time() - cache_file.stat().st_mtime
                if age < self.cache_ttl:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _set_cache(self, key: str, data: Dict):
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(f"Cache write error: {e}")
