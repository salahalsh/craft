"""
UniProt REST API Client - Subcellular localization, topology, protein families.

Feeds Module A bits (membrane topology & localization).

API: https://rest.uniprot.org/uniprotkb/
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


def _comment_type(comment: Dict) -> str:
    """UniProt REST JSON writes comment types with spaces ('SUBCELLULAR
    LOCATION'); accept the underscore form too."""
    return (comment.get('commentType') or '').replace('_', ' ').upper()


@dataclass
class UniProtTargetData:
    """Structured target data from UniProt."""
    uniprot_id: str = ''
    protein_name: str = ''
    gene_name: str = ''
    organism: str = ''
    subcellular_locations: List[str] = field(default_factory=list)
    transmembrane_count: int = 0
    topology_description: str = ''      # 'single-pass', 'multi-pass', 'soluble'
    signal_peptide: bool = False
    gpi_anchor: bool = False
    protein_families: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    go_cellular_component: List[str] = field(default_factory=list)
    tissue_specificity: str = ''
    function_description: str = ''
    sequence_length: int = 0
    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


class UniProtClient:
    """UniProt REST API v2 client with caching."""

    BASE_URL = 'https://rest.uniprot.org/uniprotkb'

    def __init__(self, cache_dir: Optional[Path] = None, cache_ttl_hours: Optional[int] = None):
        if cache_dir is None:
            from craft import config
            cache_dir = config.cache_dir('uniprot')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from craft import config
        self.cache_ttl = (config.cache_ttl_hours() if cache_ttl_hours is None
                          else cache_ttl_hours) * 3600
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'InsilicoSigma-CRAFT/1.0 (https://insilicosigma.com)',
        })
        self._last_request_time = 0
        self._min_interval = 0.1  # UniProt allows 100 req/sec unauthenticated

    def close(self):
        """Close the underlying HTTP session to release connections."""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __del__(self):
        self.close()

    def fetch(self, uniprot_id: str) -> UniProtTargetData:
        """Fetch target data by UniProt accession."""
        if not uniprot_id:
            return UniProtTargetData(success=False, error="Empty UniProt ID")

        # 'v2:' = parser fixed in v0.3.0; records cached by the old parser
        # (empty localization and tissue fields) are not reused.
        cached = self._get_cache(f"v2:{uniprot_id}")
        if cached:
            return UniProtTargetData(**cached)

        data = self._api_get(f'/{uniprot_id}')
        if data is None:
            return UniProtTargetData(
                uniprot_id=uniprot_id,
                success=False,
                error=f"UniProt entry {uniprot_id} not found")

        result = self._parse_entry(data)
        self._set_cache(f"v2:{uniprot_id}", result.to_dict())
        return result

    def _parse_entry(self, data: Dict) -> UniProtTargetData:
        """Parse UniProt JSON entry into structured data."""
        result = UniProtTargetData()
        result.uniprot_id = data.get('primaryAccession', '')

        # Protein name
        protein_desc = data.get('proteinDescription', {})
        rec_name = protein_desc.get('recommendedName', {})
        if rec_name:
            result.protein_name = rec_name.get('fullName', {}).get('value', '')
        elif protein_desc.get('submittedName'):
            result.protein_name = protein_desc['submittedName'][0].get(
                'fullName', {}).get('value', '')

        # Gene name
        genes = data.get('genes', [])
        if genes:
            result.gene_name = genes[0].get('geneName', {}).get('value', '')

        # Organism
        organism = data.get('organism', {})
        result.organism = organism.get('scientificName', '')
        result.sequence_length = int((data.get('sequence') or {}).get('length') or 0)

        # Subcellular localization
        result.subcellular_locations = self._extract_subcellular_locations(data)

        # Transmembrane topology
        features = data.get('features', [])
        tm_features = [f for f in features if f.get('type') == 'Transmembrane']
        result.transmembrane_count = len(tm_features)

        if result.transmembrane_count == 0:
            result.topology_description = 'soluble'
        elif result.transmembrane_count == 1:
            result.topology_description = 'single-pass'
        elif result.transmembrane_count == 7:
            result.topology_description = '7TM'
        else:
            result.topology_description = 'multi-pass'

        # Signal peptide
        signal_features = [f for f in features if f.get('type') == 'Signal']
        result.signal_peptide = len(signal_features) > 0

        # GPI anchor
        lipid_features = [f for f in features if f.get('type') == 'Lipidation']
        for lf in lipid_features:
            desc = lf.get('description', '').lower()
            if 'gpi' in desc:
                result.gpi_anchor = True

        # Protein families (from comments)
        comments = data.get('comments', [])
        for comment in comments:
            ctype = _comment_type(comment)
            if ctype == 'SIMILARITY':
                texts = comment.get('texts', [])
                for t in texts:
                    val = t.get('value', '')
                    if val:
                        result.protein_families.append(val)

            elif ctype == 'TISSUE SPECIFICITY':
                texts = comment.get('texts', [])
                for t in texts:
                    result.tissue_specificity += t.get('value', '') + ' '

            elif ctype == 'FUNCTION':
                texts = comment.get('texts', [])
                for t in texts:
                    result.function_description += t.get('value', '') + ' '

        result.tissue_specificity = result.tissue_specificity.strip()
        result.function_description = result.function_description.strip()

        # Keywords
        keywords = data.get('keywords', [])
        result.keywords = [kw.get('name', '') for kw in keywords]

        # GO terms (cellular component)
        cross_refs = data.get('uniProtKBCrossReferences', [])
        for ref in cross_refs:
            if ref.get('database') == 'GO':
                props = ref.get('properties', [])
                for prop in props:
                    if prop.get('key') == 'GoTerm' and prop.get('value', '').startswith('C:'):
                        result.go_cellular_component.append(
                            prop['value'][2:])  # Strip 'C:' prefix

        return result

    def _extract_subcellular_locations(self, data: Dict) -> List[str]:
        """Extract subcellular location annotations."""
        locations = []
        comments = data.get('comments', [])
        for comment in comments:
            if _comment_type(comment) == 'SUBCELLULAR LOCATION':
                subcell_locs = comment.get('subcellularLocations', [])
                for loc in subcell_locs:
                    location = loc.get('location', {})
                    val = location.get('value', '')
                    if val and val not in locations:
                        locations.append(val)
                    # Also check topology
                    topo = loc.get('topology', {})
                    topo_val = topo.get('value', '')
                    if topo_val and topo_val not in locations:
                        locations.append(topo_val)
        return locations

    def _api_get(self, endpoint: str) -> Optional[Dict]:
        """Rate-limited GET request to UniProt REST API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(3):
            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, timeout=30)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 429:
                    wait = 2 ** attempt * 5
                    logger.warning(f"UniProt rate limited, waiting {wait}s")
                    time.sleep(wait)
                else:
                    logger.warning(f"UniProt API {resp.status_code}: {url}")
                    return None

            except requests.RequestException as e:
                logger.warning(f"UniProt API error (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)

        return None

    def _get_cache(self, key: str) -> Optional[Dict]:
        """Read from disk cache."""
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
