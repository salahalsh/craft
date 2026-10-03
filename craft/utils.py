"""
CRAFT Utilities - Target identifier resolution and input parsing.
"""

import re
import csv
import io
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Regex patterns for target identifier types
UNIPROT_PATTERN = re.compile(
    r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'
)
CHEMBL_PATTERN = re.compile(r'^CHEMBL\d+$', re.IGNORECASE)


def classify_identifier(identifier: str) -> Dict[str, str]:
    """
    Classify a target identifier and return its type.

    Supports:
    - UniProt ID (e.g., P07550, Q9Y5S9, A0A024R1R8)
    - ChEMBL target ID (e.g., CHEMBL210)
    - Gene name (everything else, e.g., ADRB2, EGFR, ACE2)

    Returns:
        {'type': 'uniprot'|'chembl'|'gene', 'value': <cleaned identifier>}
    """
    cleaned = identifier.strip()
    if not cleaned:
        return {'type': 'unknown', 'value': ''}

    upper = cleaned.upper()

    # Check ChEMBL ID first (starts with CHEMBL)
    if CHEMBL_PATTERN.match(upper):
        return {'type': 'chembl', 'value': upper}

    # Check UniProt ID (6-10 alphanumeric, specific pattern)
    if UNIPROT_PATTERN.match(upper):
        return {'type': 'uniprot', 'value': upper}

    # Default to gene name
    return {'type': 'gene', 'value': upper}


def parse_target_text_input(text: str) -> List[Dict[str, str]]:
    """
    Parse text input containing target identifiers.

    Accepts:
    - One identifier per line
    - Comma-separated identifiers
    - Tab-separated identifiers
    - Mixed formats

    Returns:
        List of {'identifier': str, 'type': str} dicts
    """
    targets = []
    seen = set()

    # Split by newlines first, then by commas/tabs within each line
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # Split by comma, tab, or semicolon
        parts = re.split(r'[,;\t]+', line)
        for part in parts:
            identifier = part.strip()
            if identifier and identifier not in seen:
                info = classify_identifier(identifier)
                if info['type'] != 'unknown':
                    targets.append({
                        'identifier': info['value'],
                        'type': info['type'],
                    })
                    seen.add(info['value'])

    return targets


def parse_target_csv_file(file_content: str) -> List[Dict[str, str]]:
    """
    Parse CSV file containing target identifiers.

    Detects columns by header name (case-insensitive):
    - target_id, uniprot_id, uniprot, chembl_id, chembl, gene, gene_name, symbol, target

    Returns:
        List of {'identifier': str, 'type': str, 'name': str (optional)} dicts
    """
    targets = []
    seen = set()

    reader = csv.DictReader(io.StringIO(file_content))
    if reader.fieldnames is None:
        return targets

    # Find the identifier column
    id_column = None
    name_column = None
    header_lower = {h: h for h in reader.fieldnames}
    header_lower_map = {h.lower().strip(): h for h in reader.fieldnames}

    # Priority order for identifier column
    id_candidates = [
        'uniprot_id', 'uniprot', 'target_id', 'chembl_id', 'chembl',
        'gene_name', 'gene', 'symbol', 'target', 'id', 'identifier',
    ]
    for candidate in id_candidates:
        if candidate in header_lower_map:
            id_column = header_lower_map[candidate]
            break

    # Name column
    name_candidates = ['target_name', 'name', 'protein_name', 'description']
    for candidate in name_candidates:
        if candidate in header_lower_map:
            name_column = header_lower_map[candidate]
            break

    if id_column is None:
        # Fall back to first column
        id_column = reader.fieldnames[0]

    for row in reader:
        identifier = (row.get(id_column) or '').strip()
        if identifier and identifier not in seen:
            info = classify_identifier(identifier)
            if info['type'] != 'unknown':
                entry = {
                    'identifier': info['value'],
                    'type': info['type'],
                }
                if name_column and row.get(name_column):
                    entry['name'] = row[name_column].strip()
                targets.append(entry)
                seen.add(info['value'])

    return targets


def format_bit_count(count: int, total: int = 64) -> str:
    """Format bit count as fraction and percentage."""
    pct = (count / total * 100) if total > 0 else 0
    return f"{count}/{total} ({pct:.0f}%)"


def categorize_craft_results(result: Dict) -> Dict[str, Dict]:
    """
    Categorize CRAFT results into display sections.

    Returns dict of section_name -> {properties...} for template rendering.
    """
    categories = {
        'target_info': {
            'label': 'Target Information',
            'icon': 'fas fa-bullseye',
            'properties': {},
        },
        'module_a': {
            'label': 'Module A: Membrane Topology',
            'icon': 'fas fa-layer-group',
            'properties': {},
        },
        'module_b': {
            'label': 'Module B: Signal Transduction',
            'icon': 'fas fa-project-diagram',
            'properties': {},
        },
        'module_c': {
            'label': 'Module C: Binding Pocket',
            'icon': 'fas fa-hand-holding-medical',
            'properties': {},
        },
        'module_d': {
            'label': 'Module D: Endogenous Ligand',
            'icon': 'fas fa-pills',
            'properties': {},
        },
        'data_coverage': {
            'label': 'Data Source Coverage',
            'icon': 'fas fa-database',
            'properties': {},
        },
    }

    # Populate target info
    for key in ['target_id', 'uniprot_id', 'gene_name', 'target_name', 'organism', 'target_type']:
        if result.get(key):
            categories['target_info']['properties'][key] = result[key]

    # Populate module summaries
    categories['module_a']['properties']['on_bits'] = format_bit_count(
        result.get('module_a_count', 0))
    categories['module_b']['properties']['on_bits'] = format_bit_count(
        result.get('module_b_count', 0))
    categories['module_c']['properties']['on_bits'] = format_bit_count(
        result.get('module_c_count', 0))
    categories['module_d']['properties']['on_bits'] = format_bit_count(
        result.get('module_d_count', 0))

    # Data coverage
    sources = ['uniprot', 'iuphar', 'pocket', 'chembl']
    for src in sources:
        key = f'{src}_data_available'
        categories['data_coverage']['properties'][src] = result.get(key, False)

    return categories
