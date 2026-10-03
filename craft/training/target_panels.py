"""
Curated target panels for CRAFT validation framework.

All target identifiers, Tox21 assay-to-target mappings, and selectivity
panels in one place. Pure data - no logic.

ChEMBL IDs verified against ChEMBL REST API (2026-03-01).
Each chembl_id confirmed to map to the stated uniprot_id.
"""

# ---------------------------------------------------------------------------
# Task 1 + Task 4: Binding Affinity / Polypharmacology Targets
# 10 diverse targets across kinases, enzymes, GPCRs, NRs with abundant
# ChEMBL bioactivity data (IC50/Ki).
# ---------------------------------------------------------------------------

BINDING_AFFINITY_TARGETS = [
    {
        'chembl_id': 'CHEMBL203',
        'uniprot_id': 'P00533',
        'gene': 'EGFR',
        'family': 'kinase',
        'name': 'Epidermal growth factor receptor',
    },
    {
        'chembl_id': 'CHEMBL1824',
        'uniprot_id': 'P04626',
        'gene': 'ERBB2',
        'family': 'kinase',
        'name': 'Receptor tyrosine-protein kinase erbB-2',
    },
    {
        'chembl_id': 'CHEMBL260',
        'uniprot_id': 'Q16539',
        'gene': 'MAPK14',
        'family': 'kinase',
        'name': 'MAP kinase p38 alpha',
    },
    {
        'chembl_id': 'CHEMBL267',
        'uniprot_id': 'P12931',
        'gene': 'SRC',
        'family': 'kinase',
        'name': 'Proto-oncogene tyrosine-protein kinase Src',
    },
    {
        'chembl_id': 'CHEMBL205',
        'uniprot_id': 'P00918',
        'gene': 'CA2',
        'family': 'enzyme',
        'name': 'Carbonic anhydrase II',
    },
    {
        'chembl_id': 'CHEMBL210',
        'uniprot_id': 'P07550',
        'gene': 'ADRB2',
        'family': 'gpcr',
        'name': 'Beta-2 adrenergic receptor',
    },
    {
        'chembl_id': 'CHEMBL230',
        'uniprot_id': 'P35354',
        'gene': 'PTGS2',
        'family': 'enzyme',
        'name': 'Cyclooxygenase-2',
    },
    {
        'chembl_id': 'CHEMBL4040',
        'uniprot_id': 'P28482',
        'gene': 'MAPK1',
        'family': 'kinase',
        'name': 'MAP kinase ERK2',
    },
    {
        'chembl_id': 'CHEMBL206',
        'uniprot_id': 'P03372',
        'gene': 'ESR1',
        'family': 'nuclear_receptor',
        'name': 'Estrogen receptor alpha',
    },
    {
        'chembl_id': 'CHEMBL215',
        'uniprot_id': 'P09917',
        'gene': 'ALOX5',
        'family': 'enzyme',
        'name': 'Arachidonate 5-lipoxygenase',
    },
]


# ---------------------------------------------------------------------------
# Task 2: Tox21 Assay → Target Mapping
# Maps DeepChem Tox21 task names to protein targets for CRAFT generation.
# SR-MMP has no single protein target - CYP11A1 (mitochondrial enzyme)
# used as proxy to correctly encode mitochondrial localization bits.
# ---------------------------------------------------------------------------

TOX21_TARGET_MAPPING = {
    'NR-AR': {
        'uniprot_id': 'P10275',
        'gene': 'AR',
        'description': 'Androgen Receptor',
    },
    'NR-AR-LBD': {
        'uniprot_id': 'P10275',
        'gene': 'AR',
        'description': 'Androgen Receptor (LBD)',
    },
    'NR-AhR': {
        'uniprot_id': 'P35869',
        'gene': 'AHR',
        'description': 'Aryl Hydrocarbon Receptor',
    },
    'NR-Aromatase': {
        'uniprot_id': 'P11511',
        'gene': 'CYP19A1',
        'description': 'Aromatase',
    },
    'NR-ER': {
        'uniprot_id': 'P03372',
        'gene': 'ESR1',
        'description': 'Estrogen Receptor alpha',
    },
    'NR-ER-LBD': {
        'uniprot_id': 'P03372',
        'gene': 'ESR1',
        'description': 'Estrogen Receptor alpha (LBD)',
    },
    'NR-PPAR-gamma': {
        'uniprot_id': 'P37231',
        'gene': 'PPARG',
        'description': 'Peroxisome proliferator-activated receptor gamma',
    },
    'SR-ARE': {
        'uniprot_id': 'Q16236',
        'gene': 'NFE2L2',
        'description': 'Nuclear factor erythroid 2-related factor 2 (Nrf2)',
    },
    'SR-ATAD5': {
        # Q96QE3 = ATAD5. (Q9Y2D0, used until 2026-09, is carbonic anhydrase
        # 5B / CA5B, so the ATAD5 assay was encoded as the wrong protein.)
        'uniprot_id': 'Q96QE3',
        'gene': 'ATAD5',
        'description': 'ATPase family AAA domain-containing protein 5',
    },
    'SR-HSE': {
        'uniprot_id': 'Q00613',
        'gene': 'HSF1',
        'description': 'Heat shock factor protein 1',
    },
    'SR-MMP': {
        # No single protein target - mitochondrial membrane potential assay.
        # Using CYP11A1 as proxy: mitochondrial enzyme that correctly
        # encodes mitochondrial localization bits (bit 6) in CRAFT.
        'uniprot_id': 'P05108',
        'gene': 'CYP11A1',
        'description': 'Mitochondrial membrane potential (CYP11A1 proxy)',
        'is_proxy': True,
    },
    'SR-p53': {
        'uniprot_id': 'P04637',
        'gene': 'TP53',
        'description': 'Tumor protein p53',
    },
}

# ClinTox targets (broad cytotoxicity/FDA approval - less target-specific)
CLINTOX_TARGET_MAPPING = {
    'CT_TOX': {
        'uniprot_id': None,
        'gene': None,
        'description': 'Clinical toxicity (no specific target)',
    },
    'FDA_APPROVED': {
        'uniprot_id': None,
        'gene': None,
        'description': 'FDA approval status (no specific target)',
    },
}


# ---------------------------------------------------------------------------
# Extended Binding Affinity Panel: 50 targets across 6 families
# For publication-grade validation (Phases 9-14). Supersets the 10-target
# panel above. Each target has verified ChEMBL data (>100 compounds).
# ChEMBL IDs verified 2026-03-01 via ChEMBL REST API.
# ---------------------------------------------------------------------------

EXTENDED_BINDING_TARGETS = [
    # --- Kinases (15) ---
    {'chembl_id': 'CHEMBL203',  'uniprot_id': 'P00533', 'gene': 'EGFR',   'family': 'kinase',  'name': 'Epidermal growth factor receptor'},
    {'chembl_id': 'CHEMBL1824', 'uniprot_id': 'P04626', 'gene': 'ERBB2',  'family': 'kinase',  'name': 'Receptor tyrosine-protein kinase erbB-2'},
    {'chembl_id': 'CHEMBL260',  'uniprot_id': 'Q16539', 'gene': 'MAPK14', 'family': 'kinase',  'name': 'MAP kinase p38 alpha'},
    {'chembl_id': 'CHEMBL267',  'uniprot_id': 'P12931', 'gene': 'SRC',    'family': 'kinase',  'name': 'Proto-oncogene tyrosine-protein kinase Src'},
    {'chembl_id': 'CHEMBL4040', 'uniprot_id': 'P28482', 'gene': 'MAPK1',  'family': 'kinase',  'name': 'MAP kinase ERK2'},
    {'chembl_id': 'CHEMBL5145', 'uniprot_id': 'P15056', 'gene': 'BRAF',   'family': 'kinase',  'name': 'Serine/threonine-protein kinase B-raf'},
    {'chembl_id': 'CHEMBL1862', 'uniprot_id': 'P00519', 'gene': 'ABL1',   'family': 'kinase',  'name': 'Tyrosine-protein kinase ABL1'},
    {'chembl_id': 'CHEMBL2971', 'uniprot_id': 'O60674', 'gene': 'JAK2',   'family': 'kinase',  'name': 'Tyrosine-protein kinase JAK2'},
    {'chembl_id': 'CHEMBL301',  'uniprot_id': 'P24941', 'gene': 'CDK2',   'family': 'kinase',  'name': 'Cyclin-dependent kinase 2'},
    {'chembl_id': 'CHEMBL4247', 'uniprot_id': 'Q9UM73', 'gene': 'ALK',    'family': 'kinase',  'name': 'ALK tyrosine kinase receptor'},
    {'chembl_id': 'CHEMBL3717', 'uniprot_id': 'P08581', 'gene': 'MET',    'family': 'kinase',  'name': 'Hepatocyte growth factor receptor'},
    {'chembl_id': 'CHEMBL279',  'uniprot_id': 'P35968', 'gene': 'KDR',    'family': 'kinase',  'name': 'VEGF receptor 2'},
    {'chembl_id': 'CHEMBL4722', 'uniprot_id': 'O14965', 'gene': 'AURKA',  'family': 'kinase',  'name': 'Aurora kinase A'},
    {'chembl_id': 'CHEMBL4005', 'uniprot_id': 'P42336', 'gene': 'PIK3CA', 'family': 'kinase',  'name': 'PI3-kinase p110-alpha'},
    {'chembl_id': 'CHEMBL3650', 'uniprot_id': 'P11362', 'gene': 'FGFR1',  'family': 'kinase',  'name': 'Fibroblast growth factor receptor 1'},

    # --- GPCRs (10) ---
    {'chembl_id': 'CHEMBL210',  'uniprot_id': 'P07550', 'gene': 'ADRB2',  'family': 'gpcr',    'name': 'Beta-2 adrenergic receptor'},
    {'chembl_id': 'CHEMBL217',  'uniprot_id': 'P14416', 'gene': 'DRD2',   'family': 'gpcr',    'name': 'Dopamine D2 receptor'},
    {'chembl_id': 'CHEMBL224',  'uniprot_id': 'P28223', 'gene': 'HTR2A',  'family': 'gpcr',    'name': 'Serotonin 2A receptor'},
    {'chembl_id': 'CHEMBL233',  'uniprot_id': 'P35372', 'gene': 'OPRM1',  'family': 'gpcr',    'name': 'Mu-type opioid receptor'},
    {'chembl_id': 'CHEMBL211',  'uniprot_id': 'P08172', 'gene': 'CHRM2',  'family': 'gpcr',    'name': 'Muscarinic acetylcholine receptor M2'},
    {'chembl_id': 'CHEMBL231',  'uniprot_id': 'P35367', 'gene': 'HRH1',   'family': 'gpcr',    'name': 'Histamine H1 receptor'},
    {'chembl_id': 'CHEMBL251',  'uniprot_id': 'P29274', 'gene': 'ADORA2A','family': 'gpcr',    'name': 'Adenosine A2a receptor'},
    {'chembl_id': 'CHEMBL2107', 'uniprot_id': 'P61073', 'gene': 'CXCR4',  'family': 'gpcr',    'name': 'C-X-C chemokine receptor type 4'},
    {'chembl_id': 'CHEMBL274',  'uniprot_id': 'P51681', 'gene': 'CCR5',   'family': 'gpcr',    'name': 'C-C chemokine receptor type 5'},
    {'chembl_id': 'CHEMBL4333', 'uniprot_id': 'P21453', 'gene': 'S1PR1',  'family': 'gpcr',    'name': 'Sphingosine 1-phosphate receptor 1'},

    # --- Enzymes (10) ---
    {'chembl_id': 'CHEMBL205',  'uniprot_id': 'P00918', 'gene': 'CA2',    'family': 'enzyme',  'name': 'Carbonic anhydrase II'},
    {'chembl_id': 'CHEMBL230',  'uniprot_id': 'P35354', 'gene': 'PTGS2',  'family': 'enzyme',  'name': 'Cyclooxygenase-2'},
    {'chembl_id': 'CHEMBL215',  'uniprot_id': 'P09917', 'gene': 'ALOX5',  'family': 'enzyme',  'name': 'Arachidonate 5-lipoxygenase'},
    {'chembl_id': 'CHEMBL220',  'uniprot_id': 'P22303', 'gene': 'ACHE',   'family': 'enzyme',  'name': 'Acetylcholinesterase'},
    {'chembl_id': 'CHEMBL1827', 'uniprot_id': 'O76074', 'gene': 'PDE5A',  'family': 'enzyme',  'name': 'Phosphodiesterase 5A'},
    {'chembl_id': 'CHEMBL2039', 'uniprot_id': 'P27338', 'gene': 'MAOB',   'family': 'enzyme',  'name': 'Monoamine oxidase B'},
    {'chembl_id': 'CHEMBL4685', 'uniprot_id': 'P14902', 'gene': 'IDO1',   'family': 'enzyme',  'name': 'Indoleamine 2,3-dioxygenase 1'},
    {'chembl_id': 'CHEMBL340',  'uniprot_id': 'P08684', 'gene': 'CYP3A4', 'family': 'enzyme',  'name': 'Cytochrome P450 3A4'},
    {'chembl_id': 'CHEMBL202',  'uniprot_id': 'P00374', 'gene': 'DHFR',   'family': 'enzyme',  'name': 'Dihydrofolate reductase'},
    {'chembl_id': 'CHEMBL402',  'uniprot_id': 'P04035', 'gene': 'HMGCR',  'family': 'enzyme',  'name': 'HMG-CoA reductase'},

    # --- Nuclear Receptors (8) ---
    {'chembl_id': 'CHEMBL206',  'uniprot_id': 'P03372', 'gene': 'ESR1',   'family': 'nuclear_receptor', 'name': 'Estrogen receptor alpha'},
    {'chembl_id': 'CHEMBL242',  'uniprot_id': 'Q92731', 'gene': 'ESR2',   'family': 'nuclear_receptor', 'name': 'Estrogen receptor beta'},
    {'chembl_id': 'CHEMBL1871', 'uniprot_id': 'P10275', 'gene': 'AR',     'family': 'nuclear_receptor', 'name': 'Androgen receptor'},
    {'chembl_id': 'CHEMBL235',  'uniprot_id': 'P37231', 'gene': 'PPARG',  'family': 'nuclear_receptor', 'name': 'Peroxisome proliferator-activated receptor gamma'},
    {'chembl_id': 'CHEMBL2034', 'uniprot_id': 'P04150', 'gene': 'NR3C1',  'family': 'nuclear_receptor', 'name': 'Glucocorticoid receptor'},
    {'chembl_id': 'CHEMBL1977', 'uniprot_id': 'P11473', 'gene': 'VDR',    'family': 'nuclear_receptor', 'name': 'Vitamin D receptor'},
    {'chembl_id': 'CHEMBL2055', 'uniprot_id': 'P10276', 'gene': 'RARA',   'family': 'nuclear_receptor', 'name': 'Retinoic acid receptor alpha'},
    {'chembl_id': 'CHEMBL239',  'uniprot_id': 'Q07869', 'gene': 'PPARA',  'family': 'nuclear_receptor', 'name': 'Peroxisome proliferator-activated receptor alpha'},

    # --- Ion Channels (5) ---
    {'chembl_id': 'CHEMBL1962', 'uniprot_id': 'P14867', 'gene': 'GABRA1', 'family': 'ion_channel', 'name': 'GABA-A receptor alpha-1'},
    {'chembl_id': 'CHEMBL2015', 'uniprot_id': 'Q05586', 'gene': 'GRIN1',  'family': 'ion_channel', 'name': 'NMDA receptor subunit NR1'},
    {'chembl_id': 'CHEMBL1980', 'uniprot_id': 'Q14524', 'gene': 'SCN5A',  'family': 'ion_channel', 'name': 'Sodium channel protein type 5 alpha'},
    {'chembl_id': 'CHEMBL240',  'uniprot_id': 'Q12809', 'gene': 'KCNH2',  'family': 'ion_channel', 'name': 'hERG potassium channel'},
    {'chembl_id': 'CHEMBL1940', 'uniprot_id': 'Q13936', 'gene': 'CACNA1C','family': 'ion_channel', 'name': 'L-type calcium channel alpha-1C'},

    # --- Proteases (2) ---
    {'chembl_id': 'CHEMBL1808', 'uniprot_id': 'P12821', 'gene': 'ACE',    'family': 'protease', 'name': 'Angiotensin-converting enzyme'},
    {'chembl_id': 'CHEMBL321',  'uniprot_id': 'P14780', 'gene': 'MMP9',   'family': 'protease', 'name': 'Matrix metalloproteinase-9'},
]


# ---------------------------------------------------------------------------
# Extended Selectivity Panels (per-family for multi-family Task 3)
# ---------------------------------------------------------------------------

NUCLEAR_RECEPTOR_PANEL = [
    {'gene': 'ESR1',  'uniprot_id': 'P03372', 'family': 'nuclear_receptor', 'name': 'Estrogen receptor alpha'},
    {'gene': 'ESR2',  'uniprot_id': 'Q92731', 'family': 'nuclear_receptor', 'name': 'Estrogen receptor beta'},
    {'gene': 'AR',    'uniprot_id': 'P10275', 'family': 'nuclear_receptor', 'name': 'Androgen receptor'},
    {'gene': 'PPARG', 'uniprot_id': 'P37231', 'family': 'nuclear_receptor', 'name': 'PPAR-gamma'},
    {'gene': 'NR3C1', 'uniprot_id': 'P04150', 'family': 'nuclear_receptor', 'name': 'Glucocorticoid receptor'},
    {'gene': 'VDR',   'uniprot_id': 'P11473', 'family': 'nuclear_receptor', 'name': 'Vitamin D receptor'},
    {'gene': 'RARA',  'uniprot_id': 'P10276', 'family': 'nuclear_receptor', 'name': 'Retinoic acid receptor alpha'},
    {'gene': 'PPARA', 'uniprot_id': 'Q07869', 'family': 'nuclear_receptor', 'name': 'PPAR-alpha'},
]

ENZYME_PANEL = [
    {'gene': 'CA2',    'uniprot_id': 'P00918', 'family': 'enzyme', 'name': 'Carbonic anhydrase II'},
    {'gene': 'PTGS2',  'uniprot_id': 'P35354', 'family': 'enzyme', 'name': 'Cyclooxygenase-2'},
    {'gene': 'ALOX5',  'uniprot_id': 'P09917', 'family': 'enzyme', 'name': 'Arachidonate 5-lipoxygenase'},
    {'gene': 'ACHE',   'uniprot_id': 'P22303', 'family': 'enzyme', 'name': 'Acetylcholinesterase'},
    {'gene': 'PDE5A',  'uniprot_id': 'O76074', 'family': 'enzyme', 'name': 'Phosphodiesterase 5A'},
    {'gene': 'MAOB',   'uniprot_id': 'P27338', 'family': 'enzyme', 'name': 'Monoamine oxidase B'},
    {'gene': 'IDO1',   'uniprot_id': 'P14902', 'family': 'enzyme', 'name': 'Indoleamine 2,3-dioxygenase 1'},
    {'gene': 'CYP3A4', 'uniprot_id': 'P08684', 'family': 'enzyme', 'name': 'Cytochrome P450 3A4'},
    {'gene': 'DHFR',   'uniprot_id': 'P00374', 'family': 'enzyme', 'name': 'Dihydrofolate reductase'},
    {'gene': 'HMGCR',  'uniprot_id': 'P04035', 'family': 'enzyme', 'name': 'HMG-CoA reductase'},
]

ION_CHANNEL_PANEL = [
    {'gene': 'GABRA1',  'uniprot_id': 'P14867', 'family': 'ion_channel', 'name': 'GABA-A receptor alpha-1'},
    {'gene': 'GRIN1',   'uniprot_id': 'Q05586', 'family': 'ion_channel', 'name': 'NMDA receptor NR1'},
    {'gene': 'SCN5A',   'uniprot_id': 'Q14524', 'family': 'ion_channel', 'name': 'Sodium channel type 5'},
    {'gene': 'KCNH2',   'uniprot_id': 'Q12809', 'family': 'ion_channel', 'name': 'hERG potassium channel'},
    {'gene': 'CACNA1C', 'uniprot_id': 'Q13936', 'family': 'ion_channel', 'name': 'L-type calcium channel'},
]

PROTEASE_PANEL = [
    {'gene': 'ACE',  'uniprot_id': 'P12821', 'family': 'protease', 'name': 'Angiotensin-converting enzyme'},
    {'gene': 'MMP9', 'uniprot_id': 'P14780', 'family': 'protease', 'name': 'Matrix metalloproteinase-9'},
]


# ---------------------------------------------------------------------------
# Task 3: Selectivity Profiling Panels
# 10 kinases + 10 GPCRs for within/between-family similarity analysis.
# ---------------------------------------------------------------------------

KINASE_PANEL = [
    {'gene': 'EGFR',  'uniprot_id': 'P00533', 'name': 'Epidermal growth factor receptor'},
    {'gene': 'BRAF',  'uniprot_id': 'P15056', 'name': 'Serine/threonine-protein kinase B-raf'},
    {'gene': 'CDK2',  'uniprot_id': 'P24941', 'name': 'Cyclin-dependent kinase 2'},
    {'gene': 'ABL1',  'uniprot_id': 'P00519', 'name': 'Tyrosine-protein kinase ABL1'},
    {'gene': 'JAK2',  'uniprot_id': 'O60674', 'name': 'Tyrosine-protein kinase JAK2'},
    {'gene': 'KDR',   'uniprot_id': 'P35968', 'name': 'Vascular endothelial growth factor receptor 2'},
    {'gene': 'ALK',   'uniprot_id': 'Q9UM73', 'name': 'ALK tyrosine kinase receptor'},
    {'gene': 'MET',   'uniprot_id': 'P08581', 'name': 'Hepatocyte growth factor receptor'},
    {'gene': 'SRC',   'uniprot_id': 'P12931', 'name': 'Proto-oncogene tyrosine-protein kinase Src'},
    {'gene': 'AURKA', 'uniprot_id': 'O14965', 'name': 'Aurora kinase A'},
]

GPCR_PANEL = [
    {'gene': 'ADRB1', 'uniprot_id': 'P08588', 'name': 'Beta-1 adrenergic receptor'},
    {'gene': 'ADRB2', 'uniprot_id': 'P07550', 'name': 'Beta-2 adrenergic receptor'},
    {'gene': 'DRD1',  'uniprot_id': 'P21728', 'name': 'Dopamine D1 receptor'},
    {'gene': 'DRD2',  'uniprot_id': 'P14416', 'name': 'Dopamine D2 receptor'},
    {'gene': 'HTR1A', 'uniprot_id': 'P08908', 'name': 'Serotonin 1A receptor'},
    {'gene': 'HTR2A', 'uniprot_id': 'P28223', 'name': 'Serotonin 2A receptor'},
    {'gene': 'OPRM1', 'uniprot_id': 'P35372', 'name': 'Mu-type opioid receptor'},
    {'gene': 'OPRD1', 'uniprot_id': 'P41143', 'name': 'Delta-type opioid receptor'},
    {'gene': 'CHRM1', 'uniprot_id': 'P11229', 'name': 'Muscarinic acetylcholine receptor M1'},
    {'gene': 'CHRM2', 'uniprot_id': 'P08172', 'name': 'Muscarinic acetylcholine receptor M2'},
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_all_binding_targets():
    """Return the 10 binding affinity targets."""
    return BINDING_AFFINITY_TARGETS


def get_tox21_craft_identifier(task_name):
    """
    Return the UniProt ID for a Tox21 task, or None if unmappable.

    Args:
        task_name: Tox21 task name (e.g., 'NR-AR', 'SR-p53')

    Returns:
        UniProt ID string or None
    """
    mapping = TOX21_TARGET_MAPPING.get(task_name)
    if mapping:
        return mapping.get('uniprot_id')
    return None


def get_selectivity_panel(panel_name):
    """
    Return a selectivity panel.

    Args:
        panel_name: 'kinase' or 'gpcr'

    Returns:
        List of target dicts
    """
    if panel_name.lower() == 'kinase':
        return KINASE_PANEL
    elif panel_name.lower() == 'gpcr':
        return GPCR_PANEL
    else:
        raise ValueError(f"Unknown panel: {panel_name}. Use 'kinase' or 'gpcr'.")


def get_extended_binding_targets():
    """Return the 50-target extended panel for publication-grade validation."""
    return EXTENDED_BINDING_TARGETS


def get_targets_by_family(family):
    """
    Return targets from the extended panel filtered by family.

    Args:
        family: One of 'kinase', 'gpcr', 'enzyme', 'nuclear_receptor',
                'ion_channel', 'protease'

    Returns:
        List of target dicts
    """
    return [t for t in EXTENDED_BINDING_TARGETS if t['family'] == family]


def get_all_families():
    """Return sorted list of unique family names in the extended panel."""
    return sorted(set(t['family'] for t in EXTENDED_BINDING_TARGETS))


def get_family_panel(family):
    """
    Return the dedicated selectivity panel for a family.

    Args:
        family: Family name string

    Returns:
        List of target dicts for selectivity analysis
    """
    panels = {
        'kinase': KINASE_PANEL,
        'gpcr': GPCR_PANEL,
        'nuclear_receptor': NUCLEAR_RECEPTOR_PANEL,
        'enzyme': ENZYME_PANEL,
        'ion_channel': ION_CHANNEL_PANEL,
        'protease': PROTEASE_PANEL,
    }
    if family not in panels:
        raise ValueError(f"Unknown family: {family}. Available: {list(panels.keys())}")
    return panels[family]
