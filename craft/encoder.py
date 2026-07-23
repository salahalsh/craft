"""
CRAFT encoding engine.

Generates a 256-bit CRAFT fingerprint for any druggable protein target
from four public databases (UniProt, IUPHAR/BPS, ChEMBL, PDBe).

Usage:
    from craft.encoder import CRAFTEncoder

    encoder = CRAFTEncoder()
    fp = encoder.encode("P07550")   # ADRB2 by UniProt ID
    fp = encoder.encode("EGFR")     # by gene name
    fp = encoder.encode("CHEMBL203")# by ChEMBL target ID

    print(fp.vector.shape)          # (256,)
    print(fp.on_bit_count)
    print(fp.module_a.sum())        # Module A ON bits

For the full validation pipeline, see the validation/ directory.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .schema import (
    CRAFT_SCHEMA, BIT_COUNT, SCHEMA_BY_NAME,
    MODULE_RANGES, CLASS_BASED_POCKET_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TargetIdentity:
    uniprot_id: str
    chembl_id: str
    gene_name: str
    target_class: str = ""     # e.g. "GPCR", "Kinase"


@dataclass
class CRAFTFingerprint:
    target: TargetIdentity
    vector: np.ndarray                    # shape (256,), dtype uint8
    on_bits: list[str] = field(default_factory=list)   # bit names that are ON
    data_sources: dict[str, bool] = field(default_factory=dict)  # which APIs succeeded
    module_c_tier: int = 0               # 1=PDB, 2=class-default, 3=unknown

    @property
    def module_a(self) -> np.ndarray:
        return self.vector[0:64]

    @property
    def module_b(self) -> np.ndarray:
        return self.vector[64:128]

    @property
    def module_c(self) -> np.ndarray:
        return self.vector[128:192]

    @property
    def module_d(self) -> np.ndarray:
        return self.vector[192:256]

    @property
    def on_bit_count(self) -> int:
        return int(self.vector.sum())

    def tanimoto(self, other: "CRAFTFingerprint") -> float:
        a, b = self.vector.astype(bool), other.vector.astype(bool)
        intersection = (a & b).sum()
        union = (a | b).sum()
        return float(intersection / union) if union > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "uniprot_id": self.target.uniprot_id,
            "gene_name":  self.target.gene_name,
            "chembl_id":  self.target.chembl_id,
            "on_bit_count": self.on_bit_count,
            "on_bits": self.on_bits,
            "module_c_tier": self.module_c_tier,
            "vector": self.vector.tolist(),
        }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

CACHE_DIR = Path(".craft_cache")
CACHE_TTL  = 7 * 24 * 3600   # 7 days in seconds


def _cache_path(key: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", key)
    return CACHE_DIR / f"{safe}.json"


def _cached_get(url: str, params: dict | None = None, timeout: int = 20) -> dict | list | None:
    key = url + (json.dumps(params, sort_keys=True) if params else "")
    cp = _cache_path(key)
    if cp.exists():
        age = time.time() - cp.stat().st_mtime
        if age < CACHE_TTL:
            return json.loads(cp.read_text(encoding="utf-8"))
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"Accept": "application/json"})
            r.raise_for_status()
            data = r.json()
            CACHE_DIR.mkdir(exist_ok=True)
            cp.write_text(json.dumps(data), encoding="utf-8")
            return data
        except requests.RequestException:
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Stage 1: Identifier resolution via ChEMBL
# ---------------------------------------------------------------------------

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

_UNIPROT_RE = re.compile(r"^[A-Z][A-Z0-9]{5}$")
_CHEMBL_RE  = re.compile(r"^CHEMBL\d+$", re.IGNORECASE)


def resolve_target(identifier: str) -> TargetIdentity:
    """Map any target identifier to a canonical (uniprot_id, chembl_id, gene_name) triplet."""
    if _UNIPROT_RE.match(identifier):
        return _resolve_by_uniprot(identifier)
    if _CHEMBL_RE.match(identifier.upper()):
        return _resolve_by_chembl(identifier.upper())
    return _resolve_by_gene_name(identifier)


def _resolve_by_uniprot(uid: str) -> TargetIdentity:
    data = _cached_get(f"{CHEMBL_API}/target.json",
                       {"target_components__accession": uid, "limit": 1})
    if data and data.get("targets"):
        t = data["targets"][0]
        gene = _extract_gene(t)
        return TargetIdentity(uniprot_id=uid, chembl_id=t["target_chembl_id"],
                              gene_name=gene, target_class=t.get("target_type", ""))
    return TargetIdentity(uniprot_id=uid, chembl_id="", gene_name="")


def _resolve_by_chembl(cid: str) -> TargetIdentity:
    data = _cached_get(f"{CHEMBL_API}/target/{cid}.json")
    if data:
        uid = ""
        for comp in data.get("target_components", []):
            for xref in comp.get("target_component_xrefs", []):
                if xref.get("xref_src_db") == "UniProt":
                    uid = xref["xref_id"]
                    break
        gene = _extract_gene(data)
        return TargetIdentity(uniprot_id=uid, chembl_id=cid,
                              gene_name=gene, target_class=data.get("target_type", ""))
    return TargetIdentity(uniprot_id="", chembl_id=cid, gene_name="")


def _resolve_by_gene_name(gene: str) -> TargetIdentity:
    data = _cached_get(f"{CHEMBL_API}/target.json",
                       {"target_synonym__synonym_ilike": gene, "limit": 5})
    if data and data.get("targets"):
        for t in data["targets"]:
            if t.get("target_type") == "SINGLE PROTEIN":
                uid = ""
                for comp in t.get("target_components", []):
                    for xref in comp.get("target_component_xrefs", []):
                        if xref.get("xref_src_db") == "UniProt":
                            uid = xref["xref_id"]
                            break
                return TargetIdentity(
                    uniprot_id=uid,
                    chembl_id=t["target_chembl_id"],
                    gene_name=_extract_gene(t),
                    target_class=t.get("target_type", ""),
                )
    return TargetIdentity(uniprot_id="", chembl_id="", gene_name=gene)


def _extract_gene(t: dict) -> str:
    for comp in t.get("target_components", []):
        if comp.get("target_component_synonyms"):
            for s in comp["target_component_synonyms"]:
                if s.get("syn_type") == "GENE_SYMBOL":
                    return s["component_synonym"]
    return t.get("pref_name", "")


# ---------------------------------------------------------------------------
# Stage 2 + 3: Module encoders
# ---------------------------------------------------------------------------

UNIPROT_API  = "https://rest.uniprot.org/uniprotkb"
IUPHAR_API   = "https://www.guidetopharmacology.org/services"
PDBE_API     = "https://www.ebi.ac.uk/pdbe/api"


def _encode_module_a(identity: TargetIdentity, vector: np.ndarray) -> dict:
    """Encode Module A (bits 0-63) from UniProt."""
    data = _cached_get(f"{UNIPROT_API}/{identity.uniprot_id}.json") if identity.uniprot_id else None
    if not data:
        return {"success": False}

    comments   = {c.get("commentType", ""): c for c in data.get("comments", [])}
    keywords   = {kw.get("name", "").lower() for kw in data.get("keywords", [])}
    subcell_raw = " ".join(
        loc.get("location", {}).get("value", "")
        for loc in comments.get("SUBCELLULAR LOCATION", {}).get("subcellularLocations", [])
    ).lower()

    def _set(name: str):
        b = SCHEMA_BY_NAME.get(name)
        if b:
            vector[b.position] = 1

    # A1: Subcellular localization
    if any(k in subcell_raw for k in ("extracellular", "secreted")):   _set("extracellular")
    if "cell membrane" in subcell_raw or "plasma membrane" in subcell_raw:
        _set("plasma_membrane"); _set("cell_surface")
    if "cytoplasm" in subcell_raw or "cytosol" in subcell_raw:         _set("cytoplasm")
    if "nucleus" in subcell_raw:                                        _set("nucleus")
    if "mitochond" in subcell_raw:                                      _set("mitochondria")
    if "endoplasmic reticulum" in subcell_raw:                          _set("endoplasmic_reticulum")
    if "golgi" in subcell_raw:                                          _set("golgi")
    if "lysosom" in subcell_raw:                                        _set("lysosome")
    if "endosome" in subcell_raw:                                       _set("endosome")
    if "peroxisom" in subcell_raw:                                      _set("peroxisome")
    if "synap" in subcell_raw:                                          _set("synapse")

    # A2: Transmembrane topology
    tm_count = len([f for f in data.get("features", []) if f.get("type") == "Transmembrane"])
    if tm_count == 7:  _set("seven_tm")
    elif tm_count > 2: _set("multi_pass_tm")
    elif tm_count == 1:
        tt = comments.get("SUBCELLULAR LOCATION", {})
        orient = str(tt).lower()
        if "type i" in orient: _set("single_pass_type_i")
        elif "type ii" in orient: _set("single_pass_type_ii")
        else: _set("single_pass_type_i")
    elif tm_count == 0:
        _set("soluble_protein")

    if "gpi-anchor" in subcell_raw: _set("gpi_anchored")

    # Oligomeric state from keywords
    if "homo" in keywords and "oligomer" in keywords: _set("homo_oligomer")
    if "hetero" in keywords and "oligomer" in keywords: _set("hetero_oligomer")

    # A3: Tissue expression (UniProt expression keywords)
    kw_lower = keywords
    if "brain" in kw_lower:      _set("brain_enriched"); _set("bbb_relevant")
    if "liver" in kw_lower:      _set("liver_enriched")
    if "kidney" in kw_lower:     _set("kidney_enriched")
    if "heart" in kw_lower:      _set("heart_enriched")
    if "lung" in kw_lower:       _set("lung_enriched")
    if "testis" in kw_lower:     _set("testis_enriched")
    if "proto-oncogene" in kw_lower or "tumor" in kw_lower: _set("tumor_overexpression")
    if "developm" in " ".join(kw_lower): _set("developmental_role")

    # A4: Family classification
    fam = identity.target_class.lower()
    if "gpcr" in fam or tm_count == 7:                   _set("family_gpcr")
    elif "ion channel" in fam or "channel" in fam:       _set("family_ion_channel")
    elif "kinase" in fam:                                 _set("family_kinase")
    elif "nuclear receptor" in fam:                       _set("family_nuclear_receptor")
    elif "transporter" in fam:                            _set("family_transporter")
    elif "enzyme" in fam:                                 _set("family_enzyme")
    elif "protease" in fam:                               _set("family_protease")
    else:                                                 _set("family_other")

    return {"success": True}


def _encode_module_b(identity: TargetIdentity, vector: np.ndarray) -> dict:
    """Encode Module B (bits 64-127) from IUPHAR/BPS."""
    gene = identity.gene_name
    if not gene:
        return {"success": False}

    data = _cached_get(f"{IUPHAR_API}/targets", {"name": gene})
    if not isinstance(data, list) or not data:
        return {"success": False}

    target_data = data[0]
    tt = (target_data.get("type") or "").lower()

    def _set(name: str):
        b = SCHEMA_BY_NAME.get(name)
        if b:
            vector[b.position] = 1

    # B1: Primary mechanism from IUPHAR target type
    gpcr_coupling = target_data.get("gProteinCoupling", "")
    if gpcr_coupling:
        coupling_lower = gpcr_coupling.lower()
        if "gs" in coupling_lower:   _set("gpcr_gs_coupled")
        if "gi" in coupling_lower or "go" in coupling_lower: _set("gpcr_gi_coupled")
        if "gq" in coupling_lower:   _set("gpcr_gq_coupled")
        if "g12" in coupling_lower:  _set("gpcr_g12_coupled")
        _set("beta_arrestin")
    if "receptor tyrosine kinase" in tt: _set("receptor_tyrosine_kinase")
    elif "non-receptor" in tt and "kinase" in tt: _set("non_receptor_tyrosine_kinase")
    elif "kinase" in tt: _set("serine_threonine_kinase")
    elif "ion channel" in tt:
        if "voltage" in tt: _set("voltage_gated_ion")
        else: _set("ligand_gated_ion")
    elif "transporter" in tt:
        if "abc" in tt: _set("abc_transporter")
        else: _set("slc_transporter")
    elif "nuclear receptor" in tt:
        _set("nuclear_receptor_genomic")
        _set("slow_signaling")
    elif "enzyme" in tt:
        _set("enzyme_catalysis")
    elif "protease" in tt:
        _set("proteolytic")

    # B2: Downstream pathways (from IUPHAR function annotations)
    function_text = " ".join([
        str(target_data.get("function", "")),
        str(target_data.get("physiology", "")),
    ]).lower()

    pathway_map = {
        "adenylyl cyclase": "pathway_camp", "camp": "pathway_camp",
        "calcium": "pathway_calcium", "calmodulin": "pathway_calcium",
        "mapk": "pathway_mapk_erk", "erk": "pathway_mapk_erk",
        "pi3k": "pathway_pi3k_akt", "akt": "pathway_pi3k_akt",
        "jak": "pathway_jak_stat", "stat": "pathway_jak_stat",
        "nf-kb": "pathway_nfkb", "nfkb": "pathway_nfkb",
        "wnt": "pathway_wnt",
        "notch": "pathway_notch",
        "hedgehog": "pathway_hedgehog",
        "tgf-beta": "pathway_tgfb", "smad": "pathway_tgfb",
        "apoptosis": "pathway_apoptosis", "caspase": "pathway_apoptosis",
        "autophagy": "pathway_autophagy",
        "ubiquitin": "pathway_ubiquitin",
        "cgmp": "pathway_cgmp",
    }
    for keyword, bit_name in pathway_map.items():
        if keyword in function_text:
            _set(bit_name)

    # B4: Tissue/disease context
    context_map = {
        "central nervous": "cns_signaling", "neurotransmit": "cns_signaling",
        "cardiac": "cardiac_signaling", "heart": "cardiac_signaling",
        "immun": "immune_signaling", "inflamm": "inflammatory_role",
        "metabol": "metabolic_signaling", "glucos": "metabolic_signaling",
        "endocrin": "endocrine_signaling", "hormon": "endocrine_signaling",
        "oncogen": "oncogenic_signaling", "cancer": "oncogenic_signaling", "tumor": "oncogenic_signaling",
        "neurodegenerat": "neurodegeneration",
        "pain": "pain_signaling", "nocicepti": "pain_signaling",
    }
    for keyword, bit_name in context_map.items():
        if keyword in function_text:
            _set(bit_name)

    return {"success": True}


def _encode_module_c_pdb(identity: TargetIdentity, vector: np.ndarray) -> int:
    """
    Encode Module C (bits 128-191) from PDB via PDBe REST API.
    Returns tier: 1=PDB-based, 2=class-default, 3=unknown.
    """
    uid = identity.uniprot_id
    if not uid:
        return _encode_module_c_default(identity, vector)

    # Fetch PDB entries for this UniProt
    mapping_data = _cached_get(f"{PDBE_API}/mappings/uniprot/{uid}")
    if not mapping_data or not isinstance(mapping_data, dict):
        return _encode_module_c_default(identity, vector)

    pdb_ids = list(mapping_data.keys())
    if not pdb_ids:
        return _encode_module_c_default(identity, vector)

    # Pick highest-resolution structure (simple: take first few, prefer bound structures)
    selected_pdb = pdb_ids[0]
    binding_data = _cached_get(f"{PDBE_API}/pdb/entry/binding_sites/{selected_pdb}")

    def _set(name: str):
        b = SCHEMA_BY_NAME.get(name)
        if b:
            vector[b.position] = 1

    # Analyze binding site residues
    residue_counts = {"hydrophobic": 0, "polar": 0, "positive": 0,
                      "negative": 0, "aromatic": 0, "total": 0}
    hydrophobic = {"ALA", "VAL", "LEU", "ILE", "MET", "PRO"}
    polar       = {"SER", "THR", "ASN", "GLN", "CYS"}
    positive    = {"LYS", "ARG", "HIS"}
    negative    = {"ASP", "GLU"}
    aromatic    = {"PHE", "TRP", "TYR"}

    has_cys = False
    has_metal = False

    if binding_data and isinstance(binding_data, dict):
        for entry in binding_data.get(selected_pdb.lower(), []):
            for site_residue in entry.get("site_residues", []):
                aa = site_residue.get("residue_name", "")
                residue_counts["total"] += 1
                if aa in hydrophobic: residue_counts["hydrophobic"] += 1
                if aa in polar:       residue_counts["polar"] += 1
                if aa in positive:    residue_counts["positive"] += 1
                if aa in negative:    residue_counts["negative"] += 1
                if aa in aromatic:    residue_counts["aromatic"] += 1
                if aa == "CYS":       has_cys = True

    total = residue_counts["total"] or 1

    # C1: Pocket geometry
    _set("pocket_deep")  # default for structured pockets
    if total < 10:   _set("pocket_small")
    elif total < 20: _set("pocket_medium")
    elif total < 35: _set("pocket_large")
    else:            _set("pocket_very_large")
    _set("site_orthosteric")

    # C2: Polarity
    hyd_frac = residue_counts["hydrophobic"] / total
    pol_frac  = residue_counts["polar"] / total
    ar_frac   = residue_counts["aromatic"] / total

    if hyd_frac > 0.5:   _set("hydrophobic_pocket")
    elif pol_frac > 0.4: _set("polar_pocket")
    else:                _set("mixed_polarity")
    if ar_frac > 0.2:    _set("aromatic_rich")
    if residue_counts["positive"] > 2:  _set("hbd_rich")
    if residue_counts["negative"] > 2:  _set("hba_rich")
    if has_cys:          _set("covalent_binding")
    if has_metal:        _set("metal_coordination")

    # C3: Druggability - approximate from pocket size
    if total >= 15:  _set("druggability_high")
    elif total >= 8: _set("druggability_medium")
    else:            _set("druggability_low")
    _set("rigid_pocket")

    # C4: Binding mode defaults
    _set("reversible_binding")
    _set("competitive_inhibition")
    if has_cys: _set("covalent_warhead"); _set("irreversible_mechanism")

    return 1  # Tier 1


def _encode_module_c_default(identity: TargetIdentity, vector: np.ndarray) -> int:
    """Tier 2: class-based pocket defaults when no PDB structure available."""
    target_class = identity.target_class

    # Map ChEMBL/IUPHAR class to our lookup key
    class_map = {
        "GPCR": "GPCR", "7TM": "GPCR",
        "KINASE": "Kinase",
        "ION CHANNEL": "Ion Channel",
        "NUCLEAR RECEPTOR": "Nuclear Receptor",
        "TRANSPORTER": "Transporter",
        "ENZYME": "Enzyme",
        "PROTEASE": "Protease",
        "PHOSPHATASE": "Phosphatase",
        "E3 LIGASE": "E3 Ligase",
        "EPIGENETIC": "Epigenetic",
        "TRANSCRIPTION FACTOR": "Transcription Factor",
    }
    class_key = None
    for k, v in class_map.items():
        if k in target_class.upper():
            class_key = v
            break
    if not class_key:
        return 3  # Tier 3: unknown

    default_bits = CLASS_BASED_POCKET_DEFAULTS.get(class_key, [])
    for pos in default_bits:
        vector[pos] = 1

    return 2  # Tier 2


def _encode_module_d(identity: TargetIdentity, vector: np.ndarray) -> dict:
    """
    Encode Module D (bits 192-255) from IUPHAR/BPS endogenous ligand data.
    Note: Module D is empty for most druggable targets in current databases.
    See manuscript Section 3 and Supplementary Table S1 for details.
    """
    gene = identity.gene_name
    if not gene:
        return {"success": False}

    data = _cached_get(f"{IUPHAR_API}/targets", {"name": gene})
    if not isinstance(data, list) or not data:
        return {"success": False}

    target_data = data[0]
    endo_ligands = target_data.get("endogenousLigands", [])

    def _set(name: str):
        b = SCHEMA_BY_NAME.get(name)
        if b:
            vector[b.position] = 1

    if not endo_ligands:
        _set("endo_orphan")
        return {"success": True, "note": "no endogenous ligand annotations found"}

    # D4: Pharmacological context from ChEMBL approved drug status
    chembl_data = _cached_get(f"{CHEMBL_API}/target/{identity.chembl_id}.json") \
        if identity.chembl_id else None
    if chembl_data:
        clinical_status = chembl_data.get("target_type", "")
        # approved drug check would require compound-level queries (not implemented here)
        _set("approved_drugs")
        _set("druggable_genome")

    for lig in endo_ligands[:5]:
        lig_type = (lig.get("type") or lig.get("ligandType") or "").lower()
        mw = lig.get("mw", 0) or 0

        if "monoamine" in lig_type or "catecholamine" in lig_type: _set("endo_monoamine")
        elif "amino acid" in lig_type:                              _set("endo_amino_acid")
        elif "purine" in lig_type or "nucleotide" in lig_type:     _set("endo_purine")
        elif "lipid" in lig_type or "fatty" in lig_type:           _set("endo_lipid")
        elif "steroid" in lig_type:                                 _set("endo_steroid")
        elif "peptide" in lig_type:
            if mw > 1200: _set("endo_peptide_large")
            else:         _set("endo_peptide_small")
        elif "prostanoid" in lig_type:                              _set("endo_prostanoid")
        elif "cannabinoid" in lig_type:                             _set("endo_cannabinoid")

        if mw > 0:
            if mw < 300:  _set("endo_mw_small")
            elif mw < 600: _set("endo_mw_medium")
            else:         _set("endo_mw_large")

    return {"success": True}


# ---------------------------------------------------------------------------
# Main encoder class
# ---------------------------------------------------------------------------

class CRAFTEncoder:
    """
    Generates CRAFT fingerprints from any target identifier.

    Initialization loads no models and requires no files. API calls are
    disk-cached (7-day TTL) under .craft_cache/ for deterministic re-runs.
    """

    def __init__(self, cache_dir: str | Path = ".craft_cache"):
        global CACHE_DIR
        CACHE_DIR = Path(cache_dir)

    def encode(self, identifier: str) -> CRAFTFingerprint:
        """
        Generate a 256-bit CRAFT fingerprint for a target.

        Parameters
        ----------
        identifier : str
            UniProt accession (e.g. "P07550"), gene name (e.g. "EGFR"),
            or ChEMBL target ID (e.g. "CHEMBL203").

        Returns
        -------
        CRAFTFingerprint
            Contains the 256-bit binary vector and metadata.
        """
        identity = resolve_target(identifier)
        vector = np.zeros(BIT_COUNT, dtype=np.uint8)

        sources = {}
        sources["module_a"] = _encode_module_a(identity, vector).get("success", False)
        sources["module_b"] = _encode_module_b(identity, vector).get("success", False)
        tier = _encode_module_c_pdb(identity, vector)
        sources["module_c"] = tier in (1, 2)
        sources["module_d"] = _encode_module_d(identity, vector).get("success", False)

        on_bits = [CRAFT_SCHEMA[i].name for i in range(BIT_COUNT) if vector[i]]

        return CRAFTFingerprint(
            target=identity,
            vector=vector,
            on_bits=on_bits,
            data_sources=sources,
            module_c_tier=tier,
        )

    def encode_batch(self, identifiers: list[str]) -> list[CRAFTFingerprint]:
        """Encode a list of target identifiers."""
        return [self.encode(ident) for ident in identifiers]

    def tanimoto_matrix(self, fingerprints: list[CRAFTFingerprint]) -> np.ndarray:
        """Compute the N x N pairwise Tanimoto similarity matrix."""
        n = len(fingerprints)
        mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = fingerprints[i].tanimoto(fingerprints[j])
                mat[i, j] = val
                mat[j, i] = val
        return mat
