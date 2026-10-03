"""
CRAFT Fingerprint Generator - Core 256-bit encoding engine.

Orchestrates data retrieval from 4 API clients and maps biological
annotations to the 256-bit CRAFT vector.

Each module encoder is independent - failure of one data source
does not prevent other modules from being populated.
"""

import logging
import re

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .craft_schema import (
    CRAFT_BIT_BY_NAME, MODULE_RANGES, describe_active_bits
)

logger = logging.getLogger(__name__)


@dataclass
class CRAFTFingerprint:
    """Complete CRAFT fingerprint with metadata."""
    vector: Optional[np.ndarray] = None  # shape (256,)
    mode: str = 'binary'                 # 'binary' or 'fuzzy'
    target_id: str = ''
    uniprot_id: str = ''
    chembl_id: str = ''
    gene_name: str = ''
    target_name: str = ''
    organism: str = 'Homo sapiens'
    target_type: str = ''
    module_a_count: int = 0
    module_b_count: int = 0
    module_c_count: int = 0
    module_d_count: int = 0
    total_on_bits: int = 0
    data_coverage: Dict[str, bool] = field(default_factory=dict)
    data_completeness: float = 0.0
    module_c_tier: int = 0  # 1=PDB structure, 2=class defaults, 3=unknown
    bit_annotations: Dict[int, Dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'error': self.error,
            'target_id': self.target_id,
            'uniprot_id': self.uniprot_id,
            'chembl_id': self.chembl_id,
            'gene_name': self.gene_name,
            'target_name': self.target_name,
            'organism': self.organism,
            'target_type': self.target_type,
            'craft_vector': self.vector.tolist() if self.vector is not None else [],
            'mode': self.mode,
            'module_a_count': self.module_a_count,
            'module_b_count': self.module_b_count,
            'module_c_count': self.module_c_count,
            'module_d_count': self.module_d_count,
            'total_on_bits': self.total_on_bits,
            'data_coverage': self.data_coverage,
            'data_completeness': self.data_completeness,
            'module_c_tier': self.module_c_tier,
            'bit_annotations': self.bit_annotations,
            'warnings': self.warnings,
        }


class CRAFTGenerator:
    """
    Core CRAFT fingerprint generator.

    Retrieves data from UniProt, IUPHAR, ChEMBL, and pocket analyzer,
    then encodes the 256-bit CRAFT vector.
    """

    def __init__(self):
        from .chembl_client import ChEMBLClient
        from .uniprot_client import UniProtClient
        from .iuphar_client import IUPHARClient
        from .pocket_analyzer import PocketAnalyzer

        self.chembl = ChEMBLClient()
        self.uniprot = UniProtClient()
        self.iuphar = IUPHARClient()
        self.pocket = PocketAnalyzer()

    def generate(self, identifier: str, mode: str = 'binary') -> CRAFTFingerprint:
        """
        Generate CRAFT fingerprint for a target.

        Args:
            identifier: UniProt ID, ChEMBL target ID, or gene name
            mode: 'binary' (0/1) or 'fuzzy' (0.0-1.0)

        Returns:
            CRAFTFingerprint with 256-element vector
        """
        dtype = np.float32 if mode == 'fuzzy' else np.uint8
        fp = CRAFTFingerprint(
            vector=np.zeros(256, dtype=dtype),
            mode=mode,
            target_id=identifier,
        )

        # Step 1: Resolve identifier via ChEMBL
        chembl_data = None
        try:
            chembl_data = self.chembl.resolve_target(identifier)
            if chembl_data.success:
                fp.chembl_id = chembl_data.chembl_id
                fp.uniprot_id = chembl_data.uniprot_id
                fp.gene_name = chembl_data.gene_name
                fp.target_name = chembl_data.target_name
                fp.organism = chembl_data.organism
                fp.target_type = chembl_data.target_class_l2 or chembl_data.target_class_l1
                fp.data_coverage['chembl'] = True
            else:
                fp.data_coverage['chembl'] = False
                fp.warnings.append(f"ChEMBL: {chembl_data.error}")
        except Exception as e:
            fp.data_coverage['chembl'] = False
            fp.warnings.append(f"ChEMBL error: {e}")
            logger.warning(f"ChEMBL fetch failed for {identifier}: {e}")

        # Step 2: Fetch UniProt data (Module A)
        uniprot_data = None
        try:
            uniprot_id = fp.uniprot_id or identifier
            uniprot_data = self.uniprot.fetch(uniprot_id)
            if uniprot_data.success:
                fp.data_coverage['uniprot'] = True
                if not fp.gene_name and uniprot_data.gene_name:
                    fp.gene_name = uniprot_data.gene_name
                if not fp.target_name and uniprot_data.protein_name:
                    fp.target_name = uniprot_data.protein_name
            else:
                fp.data_coverage['uniprot'] = False
                fp.warnings.append(f"UniProt: {uniprot_data.error}")
        except Exception as e:
            fp.data_coverage['uniprot'] = False
            fp.warnings.append(f"UniProt error: {e}")
            logger.warning(f"UniProt fetch failed: {e}")

        # Step 3: Fetch IUPHAR data (Module B + D)
        iuphar_data = None
        try:
            search_term = fp.gene_name or fp.target_name or identifier
            iuphar_data = self.iuphar.fetch_by_name(search_term)
            if iuphar_data.success:
                fp.data_coverage['iuphar'] = True
                if not fp.target_type and iuphar_data.target_type:
                    fp.target_type = iuphar_data.target_type
            else:
                fp.data_coverage['iuphar'] = False
                fp.warnings.append(f"IUPHAR: {iuphar_data.error}")
        except Exception as e:
            fp.data_coverage['iuphar'] = False
            fp.warnings.append(f"IUPHAR error: {e}")
            logger.warning(f"IUPHAR fetch failed: {e}")

        # Step 4: Pocket analysis (Module C)
        pocket_data = None
        try:
            pocket_data = self.pocket.analyze(
                target_type=fp.target_type,
                protein_families=uniprot_data.protein_families if uniprot_data else None,
                keywords=uniprot_data.keywords if uniprot_data else None,
                uniprot_id=fp.uniprot_id or '',
            )
            fp.data_coverage['pocket'] = pocket_data.source != 'unknown'
            _tier_map = {'pdb': 1, 'class_default': 2, 'unknown': 3}
            fp.module_c_tier = _tier_map.get(pocket_data.source, 3)
        except Exception as e:
            fp.data_coverage['pocket'] = False
            fp.module_c_tier = 3
            fp.warnings.append(f"Pocket analysis error: {e}")
            logger.warning(f"Pocket analysis failed: {e}")

        # Step 5: Encode all 4 modules
        self._encode_module_a(fp, uniprot_data, chembl_data)
        self._encode_module_b(fp, iuphar_data, uniprot_data)
        self._encode_module_c(fp, pocket_data)
        self._encode_module_d(fp, iuphar_data, chembl_data)

        # Step 6: Compute module counts and annotations
        a_start, a_end = MODULE_RANGES['A']
        b_start, b_end = MODULE_RANGES['B']
        c_start, c_end = MODULE_RANGES['C']
        d_start, d_end = MODULE_RANGES['D']

        if mode == 'fuzzy':
            fp.module_a_count = int((fp.vector[a_start:a_end+1] > 0).sum())
            fp.module_b_count = int((fp.vector[b_start:b_end+1] > 0).sum())
            fp.module_c_count = int((fp.vector[c_start:c_end+1] > 0).sum())
            fp.module_d_count = int((fp.vector[d_start:d_end+1] > 0).sum())
        else:
            fp.module_a_count = int(fp.vector[a_start:a_end+1].sum())
            fp.module_b_count = int(fp.vector[b_start:b_end+1].sum())
            fp.module_c_count = int(fp.vector[c_start:c_end+1].sum())
            fp.module_d_count = int(fp.vector[d_start:d_end+1].sum())

        fp.total_on_bits = fp.module_a_count + fp.module_b_count + fp.module_c_count + fp.module_d_count

        # Data completeness
        available = sum(1 for v in fp.data_coverage.values() if v)
        fp.data_completeness = available / max(len(fp.data_coverage), 1)

        # Active bit annotations
        fp.bit_annotations = {
            item['position']: item
            for item in describe_active_bits(fp.vector, mode)
        }

        return fp

    # ==========================================================================
    # Module A: Membrane Topology & Localization (bits 0-63)
    # ==========================================================================

    def _encode_module_a(self, fp, uniprot_data, chembl_data):
        """Encode Module A bits from UniProt data."""
        v = fp.vector
        is_fuzzy = fp.mode == 'fuzzy'
        high = 1.0 if is_fuzzy else 1
        medium = 0.5 if is_fuzzy else 1  # In binary mode, medium -> 1

        if uniprot_data and uniprot_data.success:
            # Sub-block A1: Subcellular localization (bits 0-15)
            loc_map = {
                'extracellular': 0, 'secreted': 0,
                'cell surface': 1,
                'cell membrane': 2, 'plasma membrane': 2,
                'peripheral membrane': 3,
                'cytoplasm': 4, 'cytosol': 4,
                'nucleus': 5, 'nucleoplasm': 5,
                'mitochondri': 6,
                'endoplasmic reticulum': 7,
                'golgi': 8,
                'lysosome': 9,
                'endosome': 10,
                'peroxisome': 11,
                'cytoskeleton': 12,
                'synapse': 13, 'presynaptic': 13, 'postsynaptic': 13,
                'cell junction': 14,
                'secretory vesicle': 15, 'synaptic vesicle': 15,
            }

            for loc in uniprot_data.subcellular_locations:
                loc_lower = loc.lower()
                for keyword, bit_pos in loc_map.items():
                    if keyword in loc_lower:
                        v[bit_pos] = high

            # Sub-block A2: Transmembrane topology (bits 16-31)
            tm = uniprot_data.transmembrane_count
            if tm == 1:
                v[16] = high  # single_pass_type_i (default)
            if tm >= 2 and tm < 7:
                v[18] = high  # multi_pass_tm
            if tm == 7:
                v[19] = high  # seven_tm (GPCR)
            if uniprot_data.gpi_anchor:
                v[20] = high
            if tm == 0 and not uniprot_data.gpi_anchor:
                v[22] = high  # soluble_protein

            # TM helix count bins
            if 1 <= tm <= 2:
                v[23] = high
            elif 3 <= tm <= 6:
                v[24] = high
            elif tm >= 7:
                v[25] = high

            # Binding site orientation (derived from topology)
            if tm >= 1:
                v[26] = high  # extracellular-facing (most receptors)
            if v[4] or v[5]:  # cytoplasm or nucleus
                v[27] = high  # intracellular-facing
            if tm >= 7:
                v[28] = medium  # within_membrane (GPCRs sometimes)

            # Oligomerization (from keywords)
            kw_text = ' '.join(uniprot_data.keywords).lower()
            if 'homodimer' in kw_text or 'homotetramer' in kw_text:
                v[29] = high
            if 'heterodimer' in kw_text or 'heteromultimer' in kw_text:
                v[30] = high
            if 'disulfide bond' in kw_text:
                v[31] = medium

            # Sub-block A3: Tissue expression (bits 32-47)
            tissue_text = (uniprot_data.tissue_specificity or '').lower()
            go_text = ' '.join(uniprot_data.go_cellular_component).lower()
            combined_text = tissue_text + ' ' + go_text

            if 'ubiquit' in combined_text or 'widely expressed' in combined_text:
                v[32] = high
            if 'brain' in combined_text or 'neuron' in combined_text or 'cns' in combined_text:
                v[33] = high
            if 'liver' in combined_text or 'hepat' in combined_text:
                v[34] = high
            if 'kidney' in combined_text or 'renal' in combined_text:
                v[35] = high
            if 'heart' in combined_text or 'cardiac' in combined_text:
                v[36] = high
            if 'lymphocyte' in combined_text or 'immune' in combined_text or 'leukocyte' in combined_text:
                v[37] = high
            if 'lung' in combined_text or 'pulmon' in combined_text:
                v[38] = high
            if 'intestin' in combined_text or 'colon' in combined_text or 'gastric' in combined_text:
                v[39] = high
            if 'endocrine' in combined_text or 'thyroid' in combined_text or 'adrenal' in combined_text:
                v[40] = high
            if 'muscle' in combined_text or 'skeletal' in combined_text:
                v[41] = high
            if 'skin' in combined_text or 'epiderm' in combined_text:
                v[42] = high
            if 'bone marrow' in combined_text or 'hematopoietic' in combined_text:
                v[43] = high

            # BBB relevance
            if v[33]:  # brain-enriched
                v[44] = high

            # Developmental role
            if 'developmental' in kw_text or 'embryo' in combined_text:
                v[47] = medium

        # Sub-block A4: Protein family (bits 48-63)
        # Use both ChEMBL classification and IUPHAR target type
        target_type = (fp.target_type or '').lower()
        if chembl_data:
            cls_text = (chembl_data.target_class_l1 + ' ' + chembl_data.target_class_l2).lower()
        else:
            cls_text = ''
        combined_cls = target_type + ' ' + cls_text

        if 'gpcr' in combined_cls or 'g protein-coupled' in combined_cls:
            v[48] = high
        if 'ion channel' in combined_cls or 'lgic' in combined_cls or 'vgic' in combined_cls:
            v[49] = high
        if 'kinase' in combined_cls:
            v[50] = high
        if 'nuclear receptor' in combined_cls or 'nuclear hormone' in combined_cls:
            v[51] = high
        if 'transporter' in combined_cls or 'carrier' in combined_cls:
            v[52] = high
        if 'enzyme' in combined_cls and 'kinase' not in combined_cls and 'protease' not in combined_cls:
            v[53] = high
        if 'protease' in combined_cls or 'peptidase' in combined_cls:
            v[54] = high
        if 'phosphatase' in combined_cls:
            v[55] = high
        if 'ubiquitin ligase' in combined_cls or 'e3 ligase' in combined_cls:
            v[56] = high
        if 'epigenetic' in combined_cls or 'hdac' in combined_cls or 'histone' in combined_cls:
            v[57] = high
        if 'chaperone' in combined_cls or re.search(r'\bhsp\d*\b', combined_cls):
            v[58] = high
        if 'structural' in combined_cls or 'tubulin' in combined_cls:
            v[59] = high
        if 'transcription factor' in combined_cls and 'nuclear receptor' not in combined_cls:
            v[60] = high
        # word-bounded: a bare 'ras' substring matched 'transferase'/'isomerase'
        if 'gtpase' in combined_cls or re.search(r'\bras\b', combined_cls):
            v[61] = high

    # ==========================================================================
    # Module B: Signal Transduction Mechanism (bits 64-127)
    # ==========================================================================

    def _encode_module_b(self, fp, iuphar_data, uniprot_data):
        """Encode Module B bits from IUPHAR and UniProt data."""
        v = fp.vector
        is_fuzzy = fp.mode == 'fuzzy'
        high = 1.0 if is_fuzzy else 1
        medium = 0.5 if is_fuzzy else 1

        if iuphar_data and iuphar_data.success:
            # Sub-block B1: Primary mechanism (bits 64-79)
            for mech in iuphar_data.transduction_mechanisms:
                mech_lower = mech.lower()
                if 'gs' in mech_lower:
                    v[64] = high
                if 'gi' in mech_lower or 'go' in mech_lower:
                    v[65] = high
                if 'gq' in mech_lower or 'g11' in mech_lower:
                    v[66] = high
                if 'g12' in mech_lower or 'g13' in mech_lower:
                    v[67] = high

            target_type_lower = (iuphar_data.target_type or '').lower()
            if 'gpcr' in target_type_lower:
                v[68] = medium  # beta_arrestin_signaling (most GPCRs)
            if 'catalytic receptor' in target_type_lower:
                v[69] = high  # receptor_tyrosine_kinase
            if 'ion channel' in target_type_lower:
                if 'lgic' in (iuphar_data.family_name or '').lower() or 'ligand' in target_type_lower:
                    v[72] = high  # ligand_gated_ion_flux
                else:
                    v[73] = high  # voltage_gated_ion_flux
            if 'transporter' in target_type_lower:
                v[74] = high  # transporter_slc (default)
            if 'nuclear receptor' in target_type_lower:
                v[76] = high  # nuclear_receptor_genomic

            # Sub-block B2: Downstream pathways (bits 80-95)
            for pathway in iuphar_data.primary_pathways:
                pw_lower = pathway.lower()
                if 'camp' in pw_lower:
                    v[80] = high
                if 'calcium' in pw_lower or 'ca2+' in pw_lower:
                    v[81] = high
                if 'ip3' in pw_lower or 'dag' in pw_lower:
                    v[94] = high

        # Derive additional pathway bits from target type
        target_type = (fp.target_type or '').lower()
        if 'kinase' in target_type:
            v[70] = medium if 'non-receptor' in target_type else 0  # non_receptor_tyrosine_kinase
            v[71] = medium  # serine_threonine_kinase (common)
            v[82] = high    # pathway_mapk_erk
            v[83] = high    # pathway_pi3k_akt

        if 'protease' in target_type or 'peptidase' in target_type:
            v[79] = high    # proteolytic_cleavage

        if 'enzyme' in target_type:
            v[78] = high    # enzymatic_catalysis

        # Sub-block B3: Signal dynamics (bits 96-111) - mostly derived
        if v[72]:  # Ion channel
            v[96] = high    # fast_signaling
        elif v[64] or v[65] or v[66]:  # GPCR
            v[97] = high    # medium_signaling
            v[99] = high    # signal_amplification
            v[100] = high   # negative_feedback
            v[101] = high   # receptor_desensitization
            v[102] = high   # receptor_internalization
        elif v[76]:  # Nuclear receptor
            v[98] = high    # slow_signaling

        # Sub-block B4: Tissue-specific signaling (bits 112-127)
        # Derive from Module A expression bits
        if fp.vector[33]:  # brain_enriched
            v[112] = high   # cns_signaling
        if fp.vector[36]:  # heart_enriched
            v[113] = high   # cardiac_signaling
        if fp.vector[37]:  # immune_enriched
            v[114] = high   # immune_signaling

    # ==========================================================================
    # Module C: Binding Pocket Character (bits 128-191)
    # ==========================================================================

    def _encode_module_c(self, fp, pocket_data):
        """Encode Module C bits from pocket analysis."""
        v = fp.vector
        is_fuzzy = fp.mode == 'fuzzy'

        if pocket_data is None or not pocket_data.success:
            return

        # Map PocketData attributes directly to bit positions
        # The bit names in craft_schema match PocketData attribute names
        pocket_bit_map = {
            'pocket_deep': 128, 'pocket_shallow': 129,
            'pocket_buried': 130, 'pocket_surface_exposed': 131,
            'pocket_vol_small': 132, 'pocket_vol_medium': 133,
            'pocket_vol_large': 134, 'pocket_vol_very_large': 135,
            'orthosteric_site': 136, 'allosteric_site_known': 137,
            'catalytic_active_site': 138, 'cryptic_pocket': 139,
            'ppi_interface': 140, 'atp_binding_site': 141,
            'substrate_channel': 142, 'covalent_binding_residue': 143,
            'pocket_hydrophobic': 144, 'pocket_polar': 145,
            'pocket_mixed': 146, 'pocket_charged_positive': 147,
            'pocket_charged_negative': 148, 'pocket_aromatic_rich': 149,
            'hbond_donor_rich': 150, 'hbond_acceptor_rich': 151,
            'metal_coordinating': 152, 'backbone_hbond': 153,
            'water_mediated': 154,
            'pocket_rigid': 160, 'pocket_semi_flexible': 161,
            'pocket_highly_flexible': 162, 'induced_fit': 163,
            'dfg_motif': 165, 'p_loop': 166, 'activation_loop': 167,
            'high_druggability': 170, 'medium_druggability': 171,
            'low_druggability': 172,
            'competitive_inhibition': 176, 'fragment_bindable': 184,
        }

        # Confidence scaling for fuzzy mode
        conf = pocket_data.confidence if is_fuzzy else 1

        for attr, bit_pos in pocket_bit_map.items():
            if getattr(pocket_data, attr, False):
                v[bit_pos] = conf if is_fuzzy else 1

    # ==========================================================================
    # Module D: Endogenous Ligand Type (bits 192-255)
    # ==========================================================================

    def _encode_module_d(self, fp, iuphar_data, chembl_data):
        """Encode Module D bits from IUPHAR endogenous ligand data."""
        v = fp.vector
        is_fuzzy = fp.mode == 'fuzzy'
        high = 1.0 if is_fuzzy else 1
        medium = 0.5 if is_fuzzy else 1

        if iuphar_data and iuphar_data.success and iuphar_data.endogenous_ligands:
            for lig in iuphar_data.endogenous_ligands:
                lig_name = lig.name.lower()
                lig_type = lig.ligand_type.lower()

                # Sub-block D1: Endogenous ligand chemical class (bits 192-215)
                # Monoamines
                monoamines = ['dopamine', 'serotonin', '5-ht', 'noradrenaline',
                              'norepinephrine', 'adrenaline', 'epinephrine',
                              'histamine', 'tyramine', 'tryptamine']
                if any(m in lig_name for m in monoamines):
                    v[192] = high

                # Amino acid neurotransmitters
                aa_nt = ['glutamate', 'gaba', 'glycine', 'aspartate', 'taurine']
                if any(a in lig_name for a in aa_nt):
                    v[193] = high

                # Acetylcholine
                if 'acetylcholine' in lig_name:
                    v[194] = high

                # Peptides
                if 'peptide' in lig_type or any(p in lig_name for p in [
                    'enkephalin', 'endorphin', 'angiotensin', 'oxytocin',
                    'vasopressin', 'bradykinin', 'substance p', 'neuropeptide',
                    'somatostatin', 'orexin', 'endothelin']):
                    v[195] = high

                # Large peptide/protein
                if any(p in lig_name for p in [
                    'insulin', 'glp-1', 'glucagon', 'epo', 'erythropoietin',
                    'growth factor', 'interferon', 'interleukin', 'cytokine']):
                    v[196] = high

                # Steroids
                if any(s in lig_name for s in [
                    'cortisol', 'estrogen', 'estradiol', 'testosterone',
                    'progesterone', 'aldosterone', 'corticoster']):
                    v[197] = high

                # Thyroid
                if any(t in lig_name for t in ['t3', 't4', 'thyroid', 'thyroxine']):
                    v[198] = high

                # Retinoid
                if any(r in lig_name for r in ['retino', 'vitamin a', 'retinoic']):
                    v[199] = high

                # Vitamin D
                if 'vitamin d' in lig_name or 'calcitriol' in lig_name:
                    v[200] = high

                # Eicosanoids
                if any(e in lig_name for e in [
                    'prostaglandin', 'pge2', 'leukotriene', 'thromboxane',
                    'prostacyclin', 'eicosanoid']):
                    v[201] = high

                # Phospholipids
                if any(p in lig_name for p in [
                    'lysophosphatid', 'sphingosine', 's1p', 'lpa',
                    'phosphatidyl']):
                    v[202] = high

                # Fatty acids / endocannabinoids
                if any(f in lig_name for f in [
                    'anandamide', '2-ag', 'fatty acid', 'oleic',
                    'arachidonic', 'endocannabinoid']):
                    v[203] = high

                # Purine nucleotides
                if any(p in lig_name for p in ['atp', 'adp', 'amp', 'utp', 'udp']):
                    v[204] = high

                # Adenosine
                if 'adenosine' in lig_name:
                    v[205] = high

            # Multiple endogenous ligands
            if len(iuphar_data.endogenous_ligands) > 1:
                v[246] = high  # multiple_endogenous
            elif len(iuphar_data.endogenous_ligands) == 1:
                v[247] = high  # single_endogenous

        elif not iuphar_data or not iuphar_data.success:
            # If no IUPHAR data, check if target type suggests orphan
            target_type = (fp.target_type or '').lower()
            if 'orphan' in target_type:
                v[215] = high  # no_known_ligand

        # Sub-block D4: Pharmacological context (bits 248-255)
        if chembl_data and chembl_data.success:
            if chembl_data.approved_drug_count and chembl_data.approved_drug_count > 0:
                v[248] = high  # approved_drug_exists
            if chembl_data.bioactivity_count and chembl_data.bioactivity_count > 100:
                v[250] = medium  # tool_compounds_available

        # Derive from ion channels
        if fp.vector[49]:  # family_ion_channel
            if not any(fp.vector[192:210]):  # No specific ligand assigned
                v[207] = high  # endo_ion_monovalent (default for channels)
