"""
Binding Pocket Analyzer - Module C bit generation.

Uses a 3-tier fallback strategy:
1. PDB-based analysis (if structure exists) - highest confidence
2. Class-based defaults from literature - medium confidence
3. Unknown bits with warning - low confidence

For the initial implementation, we focus on Tier 2 (class-based defaults)
since it covers ~80% of druggable targets without requiring 3D structures.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PocketData:
    """Binding pocket characterization data."""
    source: str = 'unknown'         # 'pdb', 'class_default', 'unknown'
    confidence: float = 0.0         # 0.0-1.0
    target_class: str = ''

    # Geometry
    pocket_deep: bool = False
    pocket_shallow: bool = False
    pocket_buried: bool = False
    pocket_surface_exposed: bool = False
    pocket_vol_small: bool = False
    pocket_vol_medium: bool = False
    pocket_vol_large: bool = False
    pocket_vol_very_large: bool = False
    orthosteric_site: bool = True   # Most drugged targets have orthosteric sites
    allosteric_site_known: bool = False
    catalytic_active_site: bool = False
    cryptic_pocket: bool = False
    ppi_interface: bool = False
    atp_binding_site: bool = False
    substrate_channel: bool = False
    covalent_binding_residue: bool = False

    # Polarity
    pocket_hydrophobic: bool = False
    pocket_polar: bool = False
    pocket_mixed: bool = False
    pocket_charged_positive: bool = False
    pocket_charged_negative: bool = False
    pocket_aromatic_rich: bool = False
    hbond_donor_rich: bool = False
    hbond_acceptor_rich: bool = False
    metal_coordinating: bool = False
    backbone_hbond: bool = False
    water_mediated: bool = False

    # Flexibility
    pocket_rigid: bool = False
    pocket_semi_flexible: bool = False
    pocket_highly_flexible: bool = False
    induced_fit: bool = False
    dfg_motif: bool = False
    p_loop: bool = False
    activation_loop: bool = False
    high_druggability: bool = False
    medium_druggability: bool = False
    low_druggability: bool = False

    # Binding mode
    competitive_inhibition: bool = True
    fragment_bindable: bool = False

    success: bool = True
    error: str = ''

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}


# Literature-derived pocket defaults for major target classes
# Sources: Barril 2013 (Drug Discovery Today), Hajduk & Greer 2007 (Nature Rev Drug Discov)
CLASS_POCKET_DEFAULTS: Dict[str, Dict[str, bool]] = {
    'GPCR': {
        'pocket_deep': True,
        'pocket_vol_large': True,
        'pocket_hydrophobic': True,
        'pocket_aromatic_rich': True,
        'pocket_semi_flexible': True,
        'induced_fit': True,
        'high_druggability': True,
        'allosteric_site_known': True,
        'water_mediated': True,
    },
    'Kinase': {
        'pocket_deep': True,
        'pocket_vol_medium': True,
        'pocket_mixed': True,
        'hbond_donor_rich': True,
        'hbond_acceptor_rich': True,
        'backbone_hbond': True,
        'atp_binding_site': True,
        'catalytic_active_site': True,
        'dfg_motif': True,
        'p_loop': True,
        'activation_loop': True,
        'pocket_semi_flexible': True,
        'high_druggability': True,
        'covalent_binding_residue': True,
        'fragment_bindable': True,
        'allosteric_site_known': True,
    },
    'Ion channel': {
        'pocket_surface_exposed': True,
        'pocket_vol_medium': True,
        'pocket_charged_positive': True,
        'pocket_polar': True,
        'pocket_semi_flexible': True,
        'medium_druggability': True,
        'allosteric_site_known': True,
    },
    'Nuclear receptor': {
        'pocket_buried': True,
        'pocket_vol_small': True,
        'pocket_hydrophobic': True,
        'pocket_rigid': True,
        'high_druggability': True,
        'induced_fit': True,
    },
    'Transporter': {
        'pocket_deep': True,
        'pocket_vol_medium': True,
        'pocket_mixed': True,
        'substrate_channel': True,
        'pocket_semi_flexible': True,
        'medium_druggability': True,
    },
    'Enzyme': {
        'catalytic_active_site': True,
        'pocket_deep': True,
        'pocket_vol_medium': True,
        'pocket_mixed': True,
        'pocket_semi_flexible': True,
        'high_druggability': True,
        'fragment_bindable': True,
    },
    'Protease': {
        'catalytic_active_site': True,
        'pocket_deep': True,
        'pocket_vol_large': True,
        'pocket_mixed': True,
        'pocket_semi_flexible': True,
        'high_druggability': True,
        'covalent_binding_residue': True,
        'fragment_bindable': True,
    },
    'Phosphatase': {
        'catalytic_active_site': True,
        'pocket_shallow': True,
        'pocket_vol_medium': True,
        'pocket_polar': True,
        'pocket_charged_positive': True,
        'metal_coordinating': True,
        'low_druggability': True,
    },
    'E3 ubiquitin ligase': {
        'ppi_interface': True,
        'pocket_surface_exposed': True,
        'pocket_vol_very_large': True,
        'pocket_shallow': True,
        'low_druggability': True,
    },
    'Epigenetic regulator': {
        'catalytic_active_site': True,
        'pocket_deep': True,
        'pocket_vol_medium': True,
        'pocket_mixed': True,
        'metal_coordinating': True,
        'high_druggability': True,
    },
    'Transcription factor': {
        'ppi_interface': True,
        'pocket_surface_exposed': True,
        'pocket_vol_very_large': True,
        'pocket_shallow': True,
        'pocket_polar': True,
        'low_druggability': True,
    },
}


class PocketAnalyzer:
    """Binding pocket characterization with class-based fallback."""

    def analyze(self, target_type: str, protein_families: Optional[List[str]] = None,
                keywords: Optional[List[str]] = None,
                uniprot_id: str = '') -> PocketData:
        """
        Analyze binding pocket using 3-tier fallback strategy.

        Tier 1: PDB-based analysis from real crystal/cryo-EM structures
        Tier 2: Class-based literature defaults (~80% of druggable targets)
        Tier 3: Generic unknown defaults

        Args:
            target_type: Target class (GPCR, Kinase, etc.)
            protein_families: UniProt protein family annotations
            keywords: UniProt keywords
            uniprot_id: UniProt accession for PDB structure lookup

        Returns:
            PocketData with populated pocket characteristics
        """
        # Tier 1: PDB-based (highest confidence)
        if uniprot_id:
            pdb_result = self._analyze_from_pdb(uniprot_id)
            if pdb_result is not None:
                pdb_result.target_class = target_type
                if keywords:
                    self._enrich_from_keywords(pdb_result, keywords)
                return pdb_result

        result = PocketData()
        result.target_class = target_type

        # Tier 2: class-based defaults
        matched_class = self._match_class(target_type, protein_families, keywords)

        if matched_class and matched_class in CLASS_POCKET_DEFAULTS:
            defaults = CLASS_POCKET_DEFAULTS[matched_class]
            for attr, value in defaults.items():
                if hasattr(result, attr):
                    setattr(result, attr, value)
            result.source = 'class_default'
            result.confidence = 0.6  # Medium confidence for class-based
        else:
            # Tier 3: unknown - set generic defaults
            result.source = 'unknown'
            result.confidence = 0.2
            result.pocket_vol_medium = True
            result.pocket_mixed = True
            result.pocket_semi_flexible = True
            result.medium_druggability = True

        # Enrich from keywords
        if keywords:
            self._enrich_from_keywords(result, keywords)

        return result

    def _analyze_from_pdb(self, uniprot_id: str) -> Optional[PocketData]:
        """
        Tier 1: Try PDB-based pocket analysis.

        Returns PocketData with source='pdb' and confidence=0.9 if a structure
        with binding site annotation is found. Returns None to fall through
        to Tier 2 if no suitable structure exists.
        """
        try:
            from .pdbe_client import PDBEClient
            client = PDBEClient()

            pdb_id = client.get_best_structure(uniprot_id)
            if not pdb_id:
                return None

            site_data = client.get_binding_site(pdb_id)
            if not site_data or not site_data.success:
                return None

            # Need meaningful residue data to be useful
            if site_data.n_residues < 3:
                return None

            return self._pdb_data_to_pocket_data(site_data)

        except Exception as e:
            logger.debug("PDB Tier 1 failed for %s: %s", uniprot_id, e)
            return None

    def _pdb_data_to_pocket_data(self, site: 'PDBBindingSiteData') -> PocketData:
        """
        Map PDB residue composition to PocketData boolean attributes.

        Thresholds derived from literature averages for druggable pockets
        (Barril 2013, Hajduk & Greer 2007).
        """
        result = PocketData()
        result.source = 'pdb'
        result.confidence = 0.9

        # Pocket size
        size = site.pocket_size_category
        if size == 'small':
            result.pocket_vol_small = True
        elif size == 'medium':
            result.pocket_vol_medium = True
        elif size == 'large':
            result.pocket_vol_large = True
        elif size == 'very_large':
            result.pocket_vol_very_large = True

        # Depth: large/very_large pockets are typically deep
        if site.n_residues >= 20:
            result.pocket_deep = True
            result.pocket_buried = True
        elif site.n_residues >= 10:
            result.pocket_deep = True
        else:
            result.pocket_shallow = True
            result.pocket_surface_exposed = True

        # Polarity from residue composition
        if site.frac_hydrophobic > 0.5:
            result.pocket_hydrophobic = True
        elif site.frac_polar + site.frac_charged > 0.5:
            result.pocket_polar = True
        else:
            result.pocket_mixed = True

        if site.frac_charged > 0.2:
            if site.n_charged_pos > site.n_charged_neg:
                result.pocket_charged_positive = True
            elif site.n_charged_neg > site.n_charged_pos:
                result.pocket_charged_negative = True

        if site.frac_aromatic > 0.2:
            result.pocket_aromatic_rich = True

        # H-bond character
        # Polar + charged residues are H-bond donors/acceptors
        if site.frac_polar + site.frac_charged > 0.3:
            result.hbond_acceptor_rich = True
        if site.n_charged_pos > 2 or site.frac_polar > 0.25:
            result.hbond_donor_rich = True

        # Special features
        if site.has_cysteine_in_site:
            result.covalent_binding_residue = True
        if site.has_metal_binding:
            result.metal_coordinating = True
        if site.has_ligand:
            result.orthosteric_site = True

        # Druggability estimate from pocket properties
        if site.n_residues >= 15 and site.frac_hydrophobic > 0.3:
            result.high_druggability = True
        elif site.n_residues >= 10:
            result.medium_druggability = True
        else:
            result.low_druggability = True

        # Default flexibility (moderate - PDB doesn't tell us directly)
        result.pocket_semi_flexible = True
        result.fragment_bindable = site.n_residues < 25

        return result

    def _match_class(self, target_type: str, protein_families: Optional[List[str]],
                     keywords: Optional[List[str]]) -> Optional[str]:
        """Match target to a known class for pocket defaults.

        v0.3.0: an empty class no longer returns early (that made Tier 2
        unreachable); the keyword and family fallbacks below are always tried.
        """
        type_lower = (target_type or '').lower()
        if type_lower:
            if 'g protein-coupled' in type_lower or 'gpcr' in type_lower:
                return 'GPCR'
            for cls in CLASS_POCKET_DEFAULTS:
                if cls.lower() in type_lower or type_lower in cls.lower():
                    return cls

        # Check keywords
        if keywords:
            keyword_lower = ' '.join(keywords).lower()
            if 'g protein-coupled receptor' in keyword_lower or 'g-protein coupled receptor' in keyword_lower:
                return 'GPCR'
            if 'kinase' in keyword_lower:
                return 'Kinase'
            if 'protease' in keyword_lower or 'peptidase' in keyword_lower:
                return 'Protease'
            if 'phosphatase' in keyword_lower:
                return 'Phosphatase'
            if 'transporter' in keyword_lower:
                return 'Transporter'
            if 'ion channel' in keyword_lower:
                return 'Ion channel'
            if 'nuclear receptor' in keyword_lower:
                return 'Nuclear receptor'
            if 'ubiquitin ligase' in keyword_lower:
                return 'E3 ubiquitin ligase'
            if 'hydrolase' in keyword_lower or 'oxidoreductase' in keyword_lower:
                return 'Enzyme'

        # Check protein families
        if protein_families:
            family_text = ' '.join(protein_families).lower()
            if 'kinase' in family_text:
                return 'Kinase'
            if 'receptor' in family_text and 'nuclear' in family_text:
                return 'Nuclear receptor'

        return None

    def _enrich_from_keywords(self, result: PocketData, keywords: List[str]):
        """Enrich pocket data from UniProt keywords."""
        kw_lower = [kw.lower() for kw in keywords]

        if 'zinc' in kw_lower or 'metal-binding' in kw_lower:
            result.metal_coordinating = True

        if 'allosteric enzyme' in kw_lower:
            result.allosteric_site_known = True
