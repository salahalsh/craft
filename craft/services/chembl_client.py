"""
ChEMBL API Client - Target resolution and classification.

Resolves any target identifier (UniProt ID, gene name, ChEMBL target ID)
to a canonical set of identifiers and target metadata.

API: https://www.ebi.ac.uk/chembl/api/data/
"""

import json
import time
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ChEMBLTargetData:
    """Resolved target information from ChEMBL."""
    chembl_id: str = ''
    uniprot_id: str = ''
    gene_name: str = ''
    target_name: str = ''
    target_type: str = ''           # e.g., 'SINGLE PROTEIN', 'PROTEIN COMPLEX'
    organism: str = 'Homo sapiens'
    target_class_l1: str = ''       # e.g., 'Enzyme', 'Membrane receptor'
    target_class_l2: str = ''       # e.g., 'Kinase', 'GPCR'
    approved_drug_count: int = 0
    bioactivity_count: int = 0
    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class ChEMBLClient:
    """ChEMBL REST API client with caching and rate limiting."""

    BASE_URL = 'https://www.ebi.ac.uk/chembl/api/data'

    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: Optional[int] = None):
        if cache_dir is None:
            from craft import config
            cache_dir = config.cache_dir('chembl')
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
        self._min_interval = 0.2  # 5 req/sec to be safe

    def close(self):
        """Close the underlying HTTP session to release connections."""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __del__(self):
        self.close()

    def resolve_target(self, identifier: str) -> ChEMBLTargetData:
        """
        Resolve any identifier to ChEMBL target data.

        Tries in order:
        1. ChEMBL target ID (CHEMBL...)
        2. UniProt ID (P07550 pattern)
        3. Gene name search
        """
        from ..utils import classify_identifier
        info = classify_identifier(identifier)

        if info['type'] == 'chembl':
            result = self._fetch_by_chembl_id(info['value'])
        elif info['type'] == 'uniprot':
            result = self._fetch_by_uniprot(info['value'])
        elif info['type'] == 'gene':
            result = self._fetch_by_gene_name(info['value'])
        else:
            return ChEMBLTargetData(success=False, error=f"Unknown identifier: {identifier}")
        if result.success and result.uniprot_id and not result.target_class_l1:
            self._attach_classification(result)
        return result

    def _attach_classification(self, result: ChEMBLTargetData) -> None:
        """Fill target_class_l1/l2 from the ChEMBL protein classification.

        The /target resource carries no class field, so the class is read from
        /target_component (by accession) -> /protein_classification, walking up
        to levels 1 and 2 (e.g. 'Enzyme' / 'Kinase', 'Membrane receptor' /
        'Family A G protein-coupled receptor').
        """
        key = f"pclass_{result.uniprot_id}"
        cached = self._get_cache(key)
        if cached is None:
            cached = self._fetch_classification(result.uniprot_id)
            if cached is None:  # network failure: do not cache, leave empty
                return
            self._set_cache(key, cached)
        result.target_class_l1 = cached.get('l1', '')
        result.target_class_l2 = cached.get('l2', '')

    def _fetch_classification(self, accession: str) -> Optional[Dict]:
        data = self._api_get('/target_component', params={'accession': accession})
        if data is None:
            return None
        comps = data.get('target_components') or []
        ids = [pc.get('protein_classification_id')
               for c in comps for pc in (c.get('protein_classifications') or [])]
        ids = [i for i in ids if i]
        if not ids:
            return {'l1': '', 'l2': ''}
        chain = []
        pid = min(ids)  # deterministic when a component has several classes
        for _ in range(10):
            node = self._api_get(f'/protein_classification/{pid}')
            if node is None:
                return None
            chain.append(node.get('pref_name') or '')
            pid = node.get('parent_id') or 0
            if not pid:
                break
        chain.reverse()
        return {'l1': chain[0] if chain else '', 'l2': chain[1] if len(chain) > 1 else ''}

    def _fetch_by_chembl_id(self, chembl_id: str) -> ChEMBLTargetData:
        """Fetch target directly by ChEMBL ID."""
        cached = self._get_cache(f"chembl_{chembl_id}")
        if cached:
            return ChEMBLTargetData(**cached)

        data = self._api_get(f'/target/{chembl_id}')
        if data is None:
            return ChEMBLTargetData(success=False, error=f"ChEMBL target {chembl_id} not found")

        result = self._parse_target_data(data)
        self._set_cache(f"chembl_{chembl_id}", result.to_dict())
        return result

    def _fetch_by_uniprot(self, uniprot_id: str) -> ChEMBLTargetData:
        """Fetch target by UniProt accession."""
        cached = self._get_cache(f"uniprot_{uniprot_id}")
        if cached:
            return ChEMBLTargetData(**cached)

        data = self._api_get(
            '/target',
            params={
                'target_components__accession': uniprot_id,
                'limit': 1,
            }
        )
        if data and data.get('targets'):
            result = self._parse_target_data(data['targets'][0])
            result.uniprot_id = uniprot_id
            self._set_cache(f"uniprot_{uniprot_id}", result.to_dict())
            return result

        # Fallback: return minimal data with just the UniProt ID
        result = ChEMBLTargetData(uniprot_id=uniprot_id)
        return result

    def _fetch_by_gene_name(self, gene_name: str) -> ChEMBLTargetData:
        """Search for target by gene name."""
        cached = self._get_cache(f"gene_v2_{gene_name}")
        if cached:
            return ChEMBLTargetData(**cached)

        data = self._api_get(
            '/target/search',
            params={'q': gene_name, 'limit': 20}
        )
        if data and data.get('targets'):
            # Rank: exact gene-symbol match, then human SINGLE PROTEIN, then
            # search order (v0.2 took the first exact match, which for 'EGFR'
            # was an EGFR/PPP1CA protein-protein-interaction target).
            def symbols(target):
                return {s.get('component_synonym', '').upper()
                        for comp in (target.get('target_components') or [])
                        for s in (comp.get('target_component_synonyms') or [])
                        if s.get('syn_type') == 'GENE_SYMBOL'}

            ranked = sorted(
                enumerate(data['targets']),
                key=lambda it: (gene_name.upper() not in symbols(it[1]),
                                it[1].get('target_type') != 'SINGLE PROTEIN',
                                it[1].get('organism') != 'Homo sapiens',
                                it[0]))
            best = ranked[0][1]

            result = self._parse_target_data(best)
            self._set_cache(f"gene_v2_{gene_name}", result.to_dict())
            return result

        return ChEMBLTargetData(
            gene_name=gene_name,
            success=False,
            error=f"No ChEMBL target found for gene '{gene_name}'")

    def _parse_target_data(self, data: Dict) -> ChEMBLTargetData:
        """Parse ChEMBL target API response into ChEMBLTargetData."""
        result = ChEMBLTargetData()
        result.chembl_id = data.get('target_chembl_id', '')
        result.target_name = data.get('pref_name', '')
        result.target_type = data.get('target_type', '')
        result.organism = data.get('organism', 'Homo sapiens')

        # Extract UniProt ID and gene name from components
        components = data.get('target_components', [])
        if components:
            comp = components[0]
            result.uniprot_id = comp.get('accession', '')

            # Gene name from synonyms
            synonyms = comp.get('target_component_synonyms', [])
            for syn in synonyms:
                if syn.get('syn_type') == 'GENE_SYMBOL':
                    result.gene_name = syn.get('component_synonym', '')
                    break

        # Target classification
        classifications = data.get('target_class', [])
        if classifications:
            cls = classifications[0]
            result.target_class_l1 = cls.get('l1', '')
            result.target_class_l2 = cls.get('l2', '')

        return result

    def _api_get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited GET request to ChEMBL API."""
        # Rate limiting
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        if params is None:
            params = {}
        params['format'] = 'json'

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
                    logger.warning(f"ChEMBL rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"ChEMBL API {resp.status_code}: {url}")
                    return None

            except requests.RequestException as e:
                logger.warning(f"ChEMBL API error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

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
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _set_cache(self, key: str, data: Dict):
        """Write to disk cache."""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError as e:
            logger.warning(f"Cache write error: {e}")
