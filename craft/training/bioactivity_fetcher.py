"""
ChEMBL Bioactivity Fetcher - Compound-target binding data retrieval.

Fetches IC50/Ki bioactivity data from ChEMBL REST API for use in
CRAFT validation tasks. Supports pagination, caching, deduplication,
and pIC50 conversion.

Separate from craft/services/chembl_client.py (target resolution only).
"""

import hashlib
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class BioactivityFetchError(RuntimeError):
    """A ChEMBL request failed after all retries.

    Raised instead of returning whatever pages were collected so far: a
    partial (or empty) result must never be mistaken for, or cached as, the
    complete bioactivity set for a target.
    """


@dataclass
class BioactivityRecord:
    """A single compound-target bioactivity measurement."""
    smiles: str = ''
    chembl_compound_id: str = ''
    target_chembl_id: str = ''
    standard_type: str = ''        # 'IC50', 'Ki', 'EC50'
    standard_value: float = 0.0    # in nM
    pIC50: float = 0.0             # -log10(value_M)
    standard_units: str = ''
    assay_chembl_id: str = ''
    data_validity: str = ''        # Empty = OK, else flagged


class ChEMBLBioactivityFetcher:
    """
    Fetch compound bioactivity data from ChEMBL REST API.

    Caches results to disk for reproducibility. Rate-limited to 5 req/sec.
    """

    BASE_URL = 'https://www.ebi.ac.uk/chembl/api/data'

    def __init__(self, cache_dir: Optional[Path] = None,
                 cache_ttl_hours: Optional[int] = None):
        if cache_dir is None:
            from craft import config
            cache_dir = config.cache_dir('bioactivity')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        from craft import config
        self.cache_ttl = (config.cache_ttl_hours() if cache_ttl_hours is None
                          else cache_ttl_hours) * 3600
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'InsilicoSigma-CRAFT-Validation/1.0',
        })
        self._last_request_time = 0
        self._min_interval = 0.2  # 5 req/sec

    def close(self):
        """Close the underlying HTTP session to release connections."""
        if hasattr(self, 'session') and self.session:
            self.session.close()

    def __del__(self):
        self.close()

    def fetch_bioactivities(
        self,
        target_chembl_id: str,
        activity_types: Optional[List[str]] = None,
        max_value_nm: float = 50000,
        limit: int = 5000,
    ) -> List[BioactivityRecord]:
        """
        Fetch bioactivity data for a target.

        Args:
            target_chembl_id: ChEMBL target ID (e.g., 'CHEMBL203')
            activity_types: Filter by type (default: ['IC50', 'Ki'])
            max_value_nm: Max activity value in nM (default: 50μM)
            limit: Max records to return per activity type

        Returns:
            Deduplicated list of BioactivityRecord objects with pIC50 values.
        """
        if activity_types is None:
            activity_types = ['IC50', 'Ki']

        # Check cache first
        cache_key = f"bioact_{target_chembl_id}_{'_'.join(activity_types)}_{max_value_nm}"
        cached = self._get_cache(cache_key)
        if cached:
            logger.info("Cache hit for %s: %d records", target_chembl_id, len(cached))
            return [BioactivityRecord(**r) for r in cached]
        if cached is not None:
            # An empty cached list is indistinguishable from a failed fetch
            # that was cached before BioactivityFetchError existed (this is how
            # EGFR and ESR1 were silently lost). Never trust it; refetch.
            logger.warning("Ignoring empty cache entry for %s; refetching",
                           target_chembl_id)

        all_records = []

        for activity_type in activity_types:
            records = self._fetch_for_type(
                target_chembl_id, activity_type, max_value_nm, limit)
            all_records.extend(records)
            logger.info("Fetched %d %s records for %s",
                        len(records), activity_type, target_chembl_id)

        # Deduplicate: per compound, keep geometric mean pIC50
        deduped = self._deduplicate(all_records)

        # Filter out invalid SMILES
        valid = [r for r in deduped if r.smiles and len(r.smiles) > 2]

        logger.info("Final: %d unique compounds for %s (from %d raw records)",
                     len(valid), target_chembl_id, len(all_records))

        # Cache result. Only reached when every page request succeeded, so the
        # cached set is complete. Empty results are not cached (see above).
        if valid:
            self._set_cache(cache_key, [self._record_to_dict(r) for r in valid])

        return valid

    def _fetch_for_type(self, target_chembl_id: str, activity_type: str,
                        max_value_nm: float, limit: int) -> List[BioactivityRecord]:
        """Fetch bioactivities of a specific type with pagination."""
        records = []
        offset = 0
        page_size = 1000  # ChEMBL max per page

        while len(records) < limit:
            params = {
                'target_chembl_id': target_chembl_id,
                'standard_type': activity_type,
                'standard_relation': '=',
                'standard_units': 'nM',
                'standard_value__lte': max_value_nm,
                'target_organism': 'Homo sapiens',
                'limit': page_size,
                'offset': offset,
            }

            data = self._api_get('/activity', params)
            if data is None:
                raise BioactivityFetchError(
                    f"ChEMBL /activity failed for {target_chembl_id} "
                    f"({activity_type}) at offset {offset} after retries; "
                    f"{len(records)} records collected so far were discarded")

            activities = data.get('activities', [])
            if not activities:
                break

            for act in activities:
                smiles = act.get('canonical_smiles', '')
                value = act.get('standard_value')
                validity = act.get('data_validity_comment', '') or ''

                # Skip flagged data
                if validity:
                    continue

                if smiles and value is not None:
                    try:
                        value_f = float(value)
                        if value_f > 0:
                            records.append(BioactivityRecord(
                                smiles=smiles,
                                chembl_compound_id=act.get('molecule_chembl_id', ''),
                                target_chembl_id=target_chembl_id,
                                standard_type=activity_type,
                                standard_value=value_f,
                                pIC50=self._to_pIC50(value_f),
                                standard_units='nM',
                                assay_chembl_id=act.get('assay_chembl_id', ''),
                                data_validity='',
                            ))
                    except (ValueError, TypeError):
                        continue

            # Check if there are more pages
            if len(activities) < page_size:
                break
            offset += page_size

        return records[:limit]

    def _deduplicate(self, records: List[BioactivityRecord]) -> List[BioactivityRecord]:
        """
        Deduplicate by compound SMILES: keep geometric mean of IC50/Ki values.

        For the same compound tested multiple times at the same target,
        the geometric mean of IC50 values is standard in cheminformatics.
        """
        compound_values = defaultdict(list)
        compound_meta = {}

        for r in records:
            key = r.smiles
            compound_values[key].append(r.standard_value)
            if key not in compound_meta:
                compound_meta[key] = r

        deduped = []
        for smiles, values in compound_values.items():
            # Geometric mean of nM values
            log_values = [math.log10(v) for v in values if v > 0]
            if not log_values:
                continue
            geomean_nm = 10 ** (sum(log_values) / len(log_values))
            pIC50 = self._to_pIC50(geomean_nm)

            meta = compound_meta[smiles]
            deduped.append(BioactivityRecord(
                smiles=smiles,
                chembl_compound_id=meta.chembl_compound_id,
                target_chembl_id=meta.target_chembl_id,
                standard_type=meta.standard_type,
                standard_value=geomean_nm,
                pIC50=pIC50,
                standard_units='nM',
                assay_chembl_id=meta.assay_chembl_id,
            ))

        return deduped

    @staticmethod
    def _to_pIC50(value_nm: float) -> float:
        """Convert nM value to pIC50 = -log10(value_M) = 9 - log10(value_nM)."""
        if value_nm <= 0:
            return 0.0
        return 9.0 - math.log10(value_nm)

    def _api_get(self, endpoint: str,
                 params: Optional[Dict] = None) -> Optional[Dict]:
        """Make a rate-limited GET request to ChEMBL API."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        url = f"{self.BASE_URL}{endpoint}"
        if params is None:
            params = {}
        params['format'] = 'json'

        # Deep pagination on large targets (EGFR has ~29k activities) is where
        # ChEMBL times out, so allow more attempts and a longer timeout.
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                self._last_request_time = time.time()
                resp = self.session.get(url, params=params, timeout=120)

                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 404:
                    return None
                elif resp.status_code == 429:
                    wait = 2 ** attempt * 5
                    logger.warning("ChEMBL rate limited, waiting %ds", wait)
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = 2 ** attempt * 3
                    logger.warning("ChEMBL server error %d, retrying in %ds",
                                   resp.status_code, wait)
                    time.sleep(wait)
                else:
                    logger.warning("ChEMBL API %d: %s", resp.status_code, url)
                    return None

            except requests.RequestException as e:
                logger.warning("ChEMBL API error (attempt %d/%d): %s",
                               attempt + 1, max_attempts, e)
                if attempt < max_attempts - 1:
                    time.sleep(5 * 2 ** attempt)

        return None

    def _get_cache(self, key: str) -> Optional[list]:
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

    def _set_cache(self, key: str, data: list):
        """Write to disk cache."""
        cache_file = self.cache_dir / f"{hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except OSError as e:
            logger.warning("Cache write error: %s", e)

    @staticmethod
    def _record_to_dict(r: BioactivityRecord) -> dict:
        """Convert BioactivityRecord to dict for caching."""
        return {
            'smiles': r.smiles,
            'chembl_compound_id': r.chembl_compound_id,
            'target_chembl_id': r.target_chembl_id,
            'standard_type': r.standard_type,
            'standard_value': r.standard_value,
            'pIC50': r.pIC50,
            'standard_units': r.standard_units,
            'assay_chembl_id': r.assay_chembl_id,
            'data_validity': r.data_validity,
        }
