"""Regression tests for the defects fixed in v0.3.0. No network access.

Run: python -m pytest tests  (or python tests/test_regressions.py)
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from craft.services.craft_schema import CRAFT_SCHEMA, MODULE_RANGES  # noqa: E402
from craft.services.pocket_analyzer import PocketAnalyzer            # noqa: E402
from craft.services.uniprot_client import UniProtClient              # noqa: E402

UNIPROT_ENTRY = {
    "primaryAccession": "P03372",
    "organism": {"scientificName": "Homo sapiens"},
    "sequence": {"length": 595},
    "comments": [
        {"commentType": "SUBCELLULAR LOCATION",
         "subcellularLocations": [{"location": {"value": "Nucleus"}},
                                  {"location": {"value": "Cytoplasm"}}]},
        {"commentType": "TISSUE SPECIFICITY",
         "texts": [{"value": "Widely expressed."}]},
    ],
    "features": [], "keywords": [], "uniProtKBCrossReferences": [],
}


def test_uniprot_comment_types_with_spaces_are_parsed():
    """v0.2 compared with 'SUBCELLULAR_LOCATION'; the API writes a space."""
    rec = UniProtClient.__new__(UniProtClient)._parse_entry(UNIPROT_ENTRY)
    assert rec.subcellular_locations == ["Nucleus", "Cytoplasm"]
    assert rec.tissue_specificity == "Widely expressed."
    assert rec.sequence_length == 595


def test_tier2_reachable_without_a_class():
    """v0.2 returned None before the keyword fallback when the class was empty."""
    pa = PocketAnalyzer.__new__(PocketAnalyzer)
    assert pa._match_class("", None, ["G protein-coupled receptor", "Membrane"]) == "GPCR"
    assert pa._match_class("", None, ["Kinase", "ATP-binding"]) == "Kinase"
    assert pa._match_class("Family A G protein-coupled receptor", None, None) == "GPCR"


def test_family_keywords_are_word_bounded():
    """'ras' must not fire inside 'transferase' or 'isomerase'."""
    src = (ROOT / "craft" / "services" / "craft_generator.py").read_text(encoding="utf-8")
    assert "'ras' in combined_cls" not in src
    assert re.search(r"\bras\b", "enzyme transferase") is None
    assert re.search(r"\bras\b", "small gtpase ras family") is not None


def _encoder_writes():
    """(module, position) for every literal `v[N] = ...` in the module encoders."""
    tree = ast.parse((ROOT / "craft" / "services" / "craft_generator.py").read_text(encoding="utf-8"))
    out = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("_encode_module_"):
            module = fn.name[-1].upper()
            for node in ast.walk(fn):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if (isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
                                and t.value.id == "v" and isinstance(t.slice, ast.Constant)):
                            out.append((module, t.slice.value, node.lineno))
    return out


def test_every_encoder_write_lands_in_its_own_module():
    """Encoders write through keyword tables, not the schema; check they agree."""
    writes = _encoder_writes()
    assert len(writes) > 50, "AST scan found too few writes; scanner broken?"
    positions = {b.position: b for b in CRAFT_SCHEMA}
    bad = []
    for module, pos, line in writes:
        lo, hi = MODULE_RANGES[module]
        if not (lo <= pos <= hi) or pos not in positions or positions[pos].module != module:
            bad.append((module, pos, line))
    assert not bad, f"encoder writes outside their module or schema: {bad}"


def test_dict_position_tables_land_in_their_module():
    """The keyword->position dicts inside each encoder (e.g. loc_map)."""
    tree = ast.parse((ROOT / "craft" / "services" / "craft_generator.py").read_text(encoding="utf-8"))
    bad = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("_encode_module_"):
            lo, hi = MODULE_RANGES[fn.name[-1].upper()]
            for node in ast.walk(fn):
                if isinstance(node, ast.Dict):
                    for val in node.values:
                        if isinstance(val, ast.Constant) and isinstance(val.value, int) \
                                and not isinstance(val.value, bool) and not (lo <= val.value <= hi):
                            bad.append((fn.name, val.value, node.lineno))
    assert not bad, bad


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
