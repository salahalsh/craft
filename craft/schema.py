"""
CRAFT 256-bit schema definition.

Every entry in CRAFT_SCHEMA defines one bit of the fingerprint:
  - position  : 0-255 (absolute bit index)
  - module    : 'A' | 'B' | 'C' | 'D'
  - sub_block : e.g. 'A1', 'B3', 'C4', 'D2'
  - name      : machine-readable snake_case identifier
  - description: human-readable biological meaning
  - source    : primary database that provides the annotation
  - category  : high-level grouping within the module

The full table (all 256 rows) is provided in Supplementary Table S1.
This file contains the authoritative Python representation for the
CRAFT encoding engine.

Reference:
  Jebril I.H., Alshehade S.A. et al.
  "CRAFT: A 256-Bit Biological Fingerprint for Drug Targets."
  Journal of Biomedical Informatics (2026).
"""

from __future__ import annotations
from dataclasses import dataclass

BIT_COUNT = 256
MODULE_SIZE = 64      # bits per module
SUB_BLOCK_SIZE = 16   # bits per sub-block (Modules A-C); Module D uses variable widths


@dataclass(frozen=True)
class BitDefinition:
    position: int       # absolute index 0-255
    module: str         # 'A', 'B', 'C', 'D'
    sub_block: str      # e.g. 'A1'
    name: str           # snake_case identifier
    description: str    # human-readable
    source: str         # e.g. 'UniProt', 'IUPHAR', 'PDB/PDBe', 'ChEMBL', 'derived'
    category: str       # high-level grouping


# ---------------------------------------------------------------------------
# Module A  (bits 0-63)  --  Membrane Topology and Localization
# ---------------------------------------------------------------------------
# A1: bits  0-15   Subcellular localization
# A2: bits 16-31   Transmembrane topology and oligomeric state
# A3: bits 32-47   Tissue expression and disease context
# A4: bits 48-63   Protein family classification

_MODULE_A = [
    # ---- A1: Subcellular localization (bits 0-15) ----
    (0,  "A", "A1", "extracellular",        "Secreted or extracellular protein",                            "UniProt", "localization"),
    (1,  "A", "A1", "cell_surface",         "Accessible from the cell surface",                             "UniProt", "localization"),
    (2,  "A", "A1", "plasma_membrane",      "Embedded in or associated with the plasma membrane",           "UniProt", "localization"),
    (3,  "A", "A1", "cytoplasm",            "Cytoplasmic / cytosolic localization",                         "UniProt", "localization"),
    (4,  "A", "A1", "cytoskeleton",         "Associated with cytoskeletal structures",                      "UniProt", "localization"),
    (5,  "A", "A1", "nucleus",              "Nuclear or nucleoplasmic localization",                        "UniProt", "localization"),
    (6,  "A", "A1", "mitochondria",         "Mitochondrial localization (inner membrane, matrix, or IMS)",  "UniProt", "localization"),
    (7,  "A", "A1", "endoplasmic_reticulum","ER lumen or ER membrane localization",                        "UniProt", "localization"),
    (8,  "A", "A1", "golgi",               "Golgi apparatus localization",                                 "UniProt", "localization"),
    (9,  "A", "A1", "lysosome",            "Lysosomal or late endosomal localization",                     "UniProt", "localization"),
    (10, "A", "A1", "endosome",            "Early or recycling endosome localization",                     "UniProt", "localization"),
    (11, "A", "A1", "peroxisome",          "Peroxisomal localization",                                     "UniProt", "localization"),
    (12, "A", "A1", "synapse",             "Pre- or post-synaptic localization",                           "UniProt", "localization"),
    (13, "A", "A1", "cell_junction",       "Located at cell junctions (tight, adherens, gap)",             "UniProt", "localization"),
    (14, "A", "A1", "secretory_vesicle",   "Packaged in secretory vesicles",                               "UniProt", "localization"),
    (15, "A", "A1", "lipid_droplet",       "Associated with lipid droplets",                               "UniProt", "localization"),
    # ---- A2: Transmembrane topology (bits 16-31) ----
    (16, "A", "A2", "single_pass_type_i",  "Single-pass type I TM (N-out, C-in)",                         "UniProt", "topology"),
    (17, "A", "A2", "single_pass_type_ii", "Single-pass type II TM (N-in, C-out)",                        "UniProt", "topology"),
    (18, "A", "A2", "multi_pass_tm",       "Multi-pass transmembrane (2-6 helices)",                      "UniProt", "topology"),
    (19, "A", "A2", "seven_tm",            "Seven-transmembrane helix (GPCR topology)",                   "UniProt", "topology"),
    (20, "A", "A2", "beta_barrel_tm",      "Beta-barrel transmembrane (e.g. porins)",                     "UniProt", "topology"),
    (21, "A", "A2", "gpi_anchored",        "Glycosylphosphatidylinositol-anchored",                       "UniProt", "topology"),
    (22, "A", "A2", "soluble_protein",     "Fully soluble, no transmembrane domain",                      "UniProt", "topology"),
    (23, "A", "A2", "peripheral_membrane", "Peripherally associated with the membrane",                   "UniProt", "topology"),
    (24, "A", "A2", "lipid_anchored",      "Lipid-modified membrane anchor (myristoylation, palmitoylation)", "UniProt", "topology"),
    (25, "A", "A2", "n_terminal_tm",       "N-terminal transmembrane signal anchor",                      "UniProt", "topology"),
    (26, "A", "A2", "binding_site_extracellular", "Primary binding site is extracellular",                "UniProt", "topology"),
    (27, "A", "A2", "binding_site_intracellular", "Primary binding site is intracellular",                "derived", "topology"),
    (28, "A", "A2", "binding_site_intramembrane", "Primary binding site is intramembrane",                "derived", "topology"),
    (29, "A", "A2", "homo_oligomer",       "Forms functional homo-oligomers",                             "UniProt", "topology"),
    (30, "A", "A2", "hetero_oligomer",     "Functions as a hetero-oligomeric complex",                    "UniProt", "topology"),
    (31, "A", "A2", "monomer",             "Functions as a monomer",                                      "UniProt", "topology"),
    # ---- A3: Tissue expression and disease context (bits 32-47) ----
    (32, "A", "A3", "ubiquitous_expression","Ubiquitously expressed across tissues",                      "UniProt", "expression"),
    (33, "A", "A3", "brain_enriched",      "Enriched or highly expressed in brain",                       "UniProt", "expression"),
    (34, "A", "A3", "liver_enriched",      "Enriched or highly expressed in liver",                       "UniProt", "expression"),
    (35, "A", "A3", "kidney_enriched",     "Enriched or highly expressed in kidney",                      "UniProt", "expression"),
    (36, "A", "A3", "heart_enriched",      "Enriched or highly expressed in heart",                       "UniProt", "expression"),
    (37, "A", "A3", "immune_enriched",     "Enriched in immune cells (T cell, B cell, macrophage)",       "UniProt", "expression"),
    (38, "A", "A3", "lung_enriched",       "Enriched or highly expressed in lung",                        "UniProt", "expression"),
    (39, "A", "A3", "gi_enriched",         "Enriched in GI tract (stomach, intestine, colon)",            "UniProt", "expression"),
    (40, "A", "A3", "endocrine_enriched",  "Enriched in endocrine tissues (pancreas, thyroid, adrenal)",  "UniProt", "expression"),
    (41, "A", "A3", "muscle_enriched",     "Enriched in skeletal or smooth muscle",                       "UniProt", "expression"),
    (42, "A", "A3", "skin_enriched",       "Enriched in skin / epidermis",                                "UniProt", "expression"),
    (43, "A", "A3", "bone_marrow_enriched","Enriched in bone marrow / hematopoietic",                     "UniProt", "expression"),
    (44, "A", "A3", "bbb_relevant",        "Brain drug target; blood-brain barrier penetration relevant", "derived", "expression"),
    (45, "A", "A3", "tumor_overexpression","Overexpressed in one or more cancer types",                   "UniProt", "expression"),
    (46, "A", "A3", "developmental_role",  "Has documented role in development / embryogenesis",          "UniProt", "expression"),
    (47, "A", "A3", "testis_enriched",     "Enriched in testis (potential reproductive side-effect risk)","UniProt", "expression"),
    # ---- A4: Protein family classification (bits 48-63) ----
    (48, "A", "A4", "family_gpcr",         "G protein-coupled receptor (class A/B/C/F)",                  "ChEMBL/IUPHAR", "family"),
    (49, "A", "A4", "family_ion_channel",  "Voltage-gated or ligand-gated ion channel",                   "ChEMBL/IUPHAR", "family"),
    (50, "A", "A4", "family_kinase",       "Protein kinase (serine/threonine or tyrosine)",                "ChEMBL",        "family"),
    (51, "A", "A4", "family_nuclear_receptor", "Nuclear hormone receptor",                                 "ChEMBL/IUPHAR", "family"),
    (52, "A", "A4", "family_transporter",  "Solute carrier (SLC) or ABC transporter",                     "ChEMBL",        "family"),
    (53, "A", "A4", "family_enzyme",       "Enzyme (not kinase or protease)",                             "ChEMBL",        "family"),
    (54, "A", "A4", "family_protease",     "Serine, cysteine, or metalloprotease",                        "ChEMBL",        "family"),
    (55, "A", "A4", "family_phosphatase",  "Protein or lipid phosphatase",                                "ChEMBL",        "family"),
    (56, "A", "A4", "family_e3_ligase",    "E3 ubiquitin ligase",                                         "ChEMBL",        "family"),
    (57, "A", "A4", "family_epigenetic",   "Epigenetic regulator (HDAC, HAT, methyltransferase)",         "ChEMBL",        "family"),
    (58, "A", "A4", "family_chaperone",    "Molecular chaperone (HSP family)",                            "ChEMBL",        "family"),
    (59, "A", "A4", "family_structural",   "Structural protein (cytoskeletal, matrix)",                   "ChEMBL",        "family"),
    (60, "A", "A4", "family_tf",           "Transcription factor (not nuclear receptor)",                 "ChEMBL",        "family"),
    (61, "A", "A4", "family_gtpase",       "GTPase / small G-protein (Ras superfamily)",                  "ChEMBL",        "family"),
    (62, "A", "A4", "family_other_receptor","Other receptor class (e.g. integrins, pattern recognition)", "ChEMBL",        "family"),
    (63, "A", "A4", "family_other",        "Other or unclassified target class",                          "ChEMBL",        "family"),
]

# ---------------------------------------------------------------------------
# Module B  (bits 64-127)  --  Signal Transduction Mechanism
# ---------------------------------------------------------------------------
# B1: bits  64-79   Primary signaling mechanism
# B2: bits  80-95   Downstream pathways and second messengers
# B3: bits  96-111  Signal dynamics and modulation
# B4: bits 112-127  Tissue-specific signaling context

_MODULE_B = [
    # ---- B1: Primary signaling mechanism (bits 64-79) ----
    (64,  "B", "B1", "gpcr_gs_coupled",   "Gs-coupled GPCR (adenylyl cyclase activation)",               "IUPHAR", "mechanism"),
    (65,  "B", "B1", "gpcr_gi_coupled",   "Gi/Go-coupled GPCR (adenylyl cyclase inhibition)",             "IUPHAR", "mechanism"),
    (66,  "B", "B1", "gpcr_gq_coupled",   "Gq/11-coupled GPCR (phospholipase C activation)",             "IUPHAR", "mechanism"),
    (67,  "B", "B1", "gpcr_g12_coupled",  "G12/13-coupled GPCR (Rho GEF activation)",                    "IUPHAR", "mechanism"),
    (68,  "B", "B1", "beta_arrestin",     "Beta-arrestin recruitment / biased agonism",                  "IUPHAR", "mechanism"),
    (69,  "B", "B1", "receptor_tyrosine_kinase", "Receptor tyrosine kinase activity",                    "IUPHAR", "mechanism"),
    (70,  "B", "B1", "non_receptor_tyrosine_kinase", "Non-receptor tyrosine kinase",                     "IUPHAR", "mechanism"),
    (71,  "B", "B1", "serine_threonine_kinase", "Serine/threonine kinase activity",                      "IUPHAR", "mechanism"),
    (72,  "B", "B1", "ligand_gated_ion",  "Ligand-gated ion channel mechanism",                          "IUPHAR", "mechanism"),
    (73,  "B", "B1", "voltage_gated_ion", "Voltage-gated ion channel mechanism",                         "IUPHAR", "mechanism"),
    (74,  "B", "B1", "slc_transporter",   "SLC family transporter (facilitated diffusion / cotransport)","IUPHAR", "mechanism"),
    (75,  "B", "B1", "abc_transporter",   "ABC family transporter (active efflux)",                      "IUPHAR", "mechanism"),
    (76,  "B", "B1", "nuclear_receptor_genomic", "Nuclear receptor genomic signaling (transcription)",   "IUPHAR", "mechanism"),
    (77,  "B", "B1", "nuclear_receptor_nongenomic", "Nuclear receptor rapid non-genomic signaling",      "IUPHAR", "mechanism"),
    (78,  "B", "B1", "enzyme_catalysis",  "Enzymatic catalysis (not kinase)",                            "IUPHAR", "mechanism"),
    (79,  "B", "B1", "proteolytic",       "Proteolytic cleavage mechanism",                              "IUPHAR", "mechanism"),
    # ---- B2: Downstream pathways and second messengers (bits 80-95) ----
    (80,  "B", "B2", "pathway_camp",      "cAMP second messenger / PKA pathway",                         "IUPHAR", "pathway"),
    (81,  "B", "B2", "pathway_calcium",   "Calcium second messenger / CaM/PKC pathway",                  "IUPHAR", "pathway"),
    (82,  "B", "B2", "pathway_mapk_erk",  "MAPK/ERK cascade",                                            "IUPHAR", "pathway"),
    (83,  "B", "B2", "pathway_pi3k_akt",  "PI3K/Akt/mTOR pathway",                                       "IUPHAR", "pathway"),
    (84,  "B", "B2", "pathway_jak_stat",  "JAK-STAT pathway",                                            "IUPHAR", "pathway"),
    (85,  "B", "B2", "pathway_nfkb",      "NF-kB pathway",                                               "IUPHAR", "pathway"),
    (86,  "B", "B2", "pathway_wnt",       "Wnt/beta-catenin pathway",                                    "IUPHAR", "pathway"),
    (87,  "B", "B2", "pathway_notch",     "Notch signaling pathway",                                     "IUPHAR", "pathway"),
    (88,  "B", "B2", "pathway_hedgehog",  "Hedgehog/Gli pathway",                                        "IUPHAR", "pathway"),
    (89,  "B", "B2", "pathway_tgfb",      "TGF-beta/SMAD pathway",                                       "IUPHAR", "pathway"),
    (90,  "B", "B2", "pathway_ddr",       "DNA damage response pathway",                                 "IUPHAR", "pathway"),
    (91,  "B", "B2", "pathway_apoptosis", "Apoptosis signaling (Bcl-2 family, caspases)",                "IUPHAR", "pathway"),
    (92,  "B", "B2", "pathway_autophagy", "Autophagy pathway (mTOR, ULK1)",                              "IUPHAR", "pathway"),
    (93,  "B", "B2", "pathway_ubiquitin", "Ubiquitin-proteasome pathway",                                "IUPHAR", "pathway"),
    (94,  "B", "B2", "pathway_ip3_dag",   "IP3/DAG second messenger system",                             "IUPHAR", "pathway"),
    (95,  "B", "B2", "pathway_cgmp",      "cGMP second messenger / PKG pathway",                         "IUPHAR", "pathway"),
    # ---- B3: Signal dynamics and modulation (bits 96-111) ----
    (96,  "B", "B3", "fast_signaling",    "Fast signal response (milliseconds - seconds)",                "IUPHAR", "dynamics"),
    (97,  "B", "B3", "medium_signaling",  "Medium signal response (seconds - minutes)",                  "IUPHAR", "dynamics"),
    (98,  "B", "B3", "slow_signaling",    "Slow signal response (minutes - hours, transcriptional)",     "IUPHAR", "dynamics"),
    (99,  "B", "B3", "signal_amplification", "Significant signal amplification at target",              "IUPHAR", "dynamics"),
    (100, "B", "B3", "negative_feedback", "Target subject to negative feedback / auto-regulation",       "IUPHAR", "dynamics"),
    (101, "B", "B3", "receptor_desensitization", "Receptor desensitization / tachyphylaxis",            "IUPHAR", "dynamics"),
    (102, "B", "B3", "receptor_internalization", "Receptor internalization / downregulation",           "IUPHAR", "dynamics"),
    (103, "B", "B3", "constitutive_activity", "Target exhibits constitutive (ligand-independent) activity", "IUPHAR", "dynamics"),
    (104, "B", "B3", "allosteric_modulation", "Target has documented allosteric modulatory sites",      "IUPHAR", "dynamics"),
    (105, "B", "B3", "biased_agonism",    "Functional selectivity / biased agonism documented",         "IUPHAR", "dynamics"),
    (106, "B", "B3", "irreversible_binding", "Target is amenable to irreversible / covalent inhibition","derived", "dynamics"),
    (107, "B", "B3", "cooperativity",     "Cooperative binding / Hill coefficient != 1",                "IUPHAR", "dynamics"),
    (108, "B", "B3", "dimerization_dependent", "Signaling depends on homo- or heterodimerization",     "IUPHAR", "dynamics"),
    (109, "B", "B3", "scaffold_protein",  "Target acts as a scaffold / adapter protein",                "UniProt", "dynamics"),
    (110, "B", "B3", "phosphorylation_site", "Target activity regulated by phosphorylation",            "UniProt", "dynamics"),
    (111, "B", "B3", "ubiquitylation_regulated", "Target activity regulated by ubiquitylation",        "UniProt", "dynamics"),
    # ---- B4: Tissue-specific signaling context (bits 112-127) ----
    (112, "B", "B4", "cns_signaling",     "Primarily or importantly involved in CNS signaling",          "IUPHAR", "context"),
    (113, "B", "B4", "cardiac_signaling", "Primarily or importantly involved in cardiac signaling",      "IUPHAR", "context"),
    (114, "B", "B4", "immune_signaling",  "Primarily or importantly involved in immune signaling",       "IUPHAR", "context"),
    (115, "B", "B4", "metabolic_signaling","Primarily or importantly involved in metabolic regulation",  "IUPHAR", "context"),
    (116, "B", "B4", "endocrine_signaling","Primarily or importantly involved in endocrine signaling",   "IUPHAR", "context"),
    (117, "B", "B4", "vascular_signaling","Primarily or importantly involved in vascular regulation",    "IUPHAR", "context"),
    (118, "B", "B4", "renal_signaling",   "Primarily or importantly involved in renal function",         "IUPHAR", "context"),
    (119, "B", "B4", "oncogenic_signaling","Target is a known oncogenic driver or frequently mutated",   "IUPHAR", "context"),
    (120, "B", "B4", "inflammatory_role", "Target has a primary role in inflammatory disease",           "IUPHAR", "context"),
    (121, "B", "B4", "neurodegeneration", "Target implicated in neurodegenerative disease",              "IUPHAR", "context"),
    (122, "B", "B4", "fibrosis_role",     "Target implicated in fibrotic disease",                       "IUPHAR", "context"),
    (123, "B", "B4", "pain_signaling",    "Target has a primary role in pain / nociception",             "IUPHAR", "context"),
    (124, "B", "B4", "circadian_role",    "Target has a role in circadian rhythm regulation",            "IUPHAR", "context"),
    (125, "B", "B4", "epigenetic_signaling","Target has a primary role in epigenetic regulation",        "IUPHAR", "context"),
    (126, "B", "B4", "reproductive_role", "Target has a primary role in reproductive function",         "IUPHAR", "context"),
    (127, "B", "B4", "developmental_signaling", "Target has a critical role in embryonic development",  "UniProt", "context"),
]

# ---------------------------------------------------------------------------
# Module C  (bits 128-191)  --  Binding Pocket Character
# ---------------------------------------------------------------------------
# C1: bits 128-143  Pocket geometry and functional site type
# C2: bits 144-159  Pocket polarity and electrostatics
# C3: bits 160-175  Pocket flexibility and kinase-specific features
# C4: bits 176-191  Binding mode and SAR characteristics

_MODULE_C = [
    # ---- C1: Pocket geometry and site type (bits 128-143) ----
    (128, "C", "C1", "pocket_deep",        "Deep binding pocket (buried)",                               "PDB/PDBe", "geometry"),
    (129, "C", "C1", "pocket_shallow",     "Shallow binding pocket (surface-exposed)",                   "PDB/PDBe", "geometry"),
    (130, "C", "C1", "pocket_buried",      "Fully buried binding site",                                  "PDB/PDBe", "geometry"),
    (131, "C", "C1", "pocket_solvent_exposed", "Partially solvent-exposed binding site",                 "PDB/PDBe", "geometry"),
    (132, "C", "C1", "pocket_small",       "Small pocket volume (<200 A3)",                              "PDB/PDBe", "geometry"),
    (133, "C", "C1", "pocket_medium",      "Medium pocket volume (200-500 A3)",                          "PDB/PDBe", "geometry"),
    (134, "C", "C1", "pocket_large",       "Large pocket volume (500-1000 A3)",                          "PDB/PDBe", "geometry"),
    (135, "C", "C1", "pocket_very_large",  "Very large pocket volume (>1000 A3)",                        "PDB/PDBe", "geometry"),
    (136, "C", "C1", "site_orthosteric",   "Primary orthosteric (endogenous) binding site",              "PDB/PDBe", "site_type"),
    (137, "C", "C1", "site_allosteric",    "Allosteric / secondary binding site",                        "PDB/PDBe", "site_type"),
    (138, "C", "C1", "site_catalytic",     "Catalytic site (enzyme active site)",                        "PDB/PDBe", "site_type"),
    (139, "C", "C1", "site_cryptic",       "Cryptic site (only visible in certain conformations)",       "PDB/PDBe", "site_type"),
    (140, "C", "C1", "site_ppi_interface", "Protein-protein interaction interface site",                 "PDB/PDBe", "site_type"),
    (141, "C", "C1", "atp_binding_site",   "ATP / nucleotide binding site (hinge region)",               "PDB/PDBe", "site_type"),
    (142, "C", "C1", "substrate_channel",  "Extended substrate binding channel",                         "PDB/PDBe", "site_type"),
    (143, "C", "C1", "covalent_binding",   "Covalent binding residue present (Cys, Ser, Tyr, Lys)",     "PDB/PDBe", "site_type"),
    # ---- C2: Pocket polarity and electrostatics (bits 144-159) ----
    (144, "C", "C2", "hydrophobic_pocket", "Predominantly hydrophobic binding environment",               "PDB/PDBe", "polarity"),
    (145, "C", "C2", "polar_pocket",       "Predominantly polar binding environment",                    "PDB/PDBe", "polarity"),
    (146, "C", "C2", "mixed_polarity",     "Mixed hydrophobic/polar binding environment",                 "PDB/PDBe", "polarity"),
    (147, "C", "C2", "positive_pocket",    "Net positively charged pocket environment",                  "PDB/PDBe", "polarity"),
    (148, "C", "C2", "negative_pocket",    "Net negatively charged pocket environment",                  "PDB/PDBe", "polarity"),
    (149, "C", "C2", "aromatic_rich",      "High aromatic residue content (Phe, Trp, Tyr)",              "PDB/PDBe", "polarity"),
    (150, "C", "C2", "hbd_rich",           "Rich in hydrogen-bond donors",                               "PDB/PDBe", "polarity"),
    (151, "C", "C2", "hba_rich",           "Rich in hydrogen-bond acceptors",                            "PDB/PDBe", "polarity"),
    (152, "C", "C2", "metal_coordination", "Metal coordination site (Zn, Mg, Fe, Ca, etc.)",             "PDB/PDBe", "polarity"),
    (153, "C", "C2", "backbone_hbond",     "Backbone hydrogen-bonding network in binding site",          "PDB/PDBe", "polarity"),
    (154, "C", "C2", "water_mediated",     "Significant water-mediated interactions in binding site",    "PDB/PDBe", "polarity"),
    (155, "C", "C2", "pi_stacking",        "Pi-stacking interactions important for binding",              "PDB/PDBe", "polarity"),
    (156, "C", "C2", "halogen_bonding",    "Halogen bonding interactions documented",                    "PDB/PDBe", "polarity"),
    (157, "C", "C2", "disulfide_near_site","Disulfide bond near or within binding site",                  "PDB/PDBe", "polarity"),
    (158, "C", "C2", "glycosylation_near", "Glycosylation site near binding pocket",                     "PDB/PDBe", "polarity"),
    (159, "C", "C2", "charged_gating",     "Electrostatic gating of binding site access",                "PDB/PDBe", "polarity"),
    # ---- C3: Pocket flexibility and kinase motifs (bits 160-175) ----
    (160, "C", "C3", "rigid_pocket",       "Rigid, geometrically defined binding site",                  "PDB/PDBe", "flexibility"),
    (161, "C", "C3", "semi_flexible",      "Semi-flexible binding site with some conformational sampling","PDB/PDBe", "flexibility"),
    (162, "C", "C3", "highly_flexible",    "Highly flexible binding site (induced fit dominant)",         "PDB/PDBe", "flexibility"),
    (163, "C", "C3", "induced_fit",        "Significant induced fit upon ligand binding documented",     "PDB/PDBe", "flexibility"),
    (164, "C", "C3", "druggability_high",  "High druggability score (Fpocket / SiteMap)",                "PDB/PDBe", "druggability"),
    (165, "C", "C3", "druggability_medium","Medium druggability score",                                  "PDB/PDBe", "druggability"),
    (166, "C", "C3", "druggability_low",   "Low druggability score",                                     "PDB/PDBe", "druggability"),
    (167, "C", "C3", "challenging_target", "Challenging / undruggable target (flat, featureless pocket)","PDB/PDBe", "druggability"),
    (168, "C", "C3", "gatekeeper_residue", "Gatekeeper residue present (kinase selectivity filter)",    "PDB/PDBe", "flexibility"),
    (169, "C", "C3", "p_loop",             "P-loop (phosphate-binding loop) in binding site",            "PDB/PDBe", "flexibility"),
    (170, "C", "C3", "activation_loop",    "Activation loop contributes to binding site",                "PDB/PDBe", "flexibility"),
    (171, "C", "C3", "hinge_region",       "Hinge region hydrogen-bond donor/acceptor (kinases)",        "PDB/PDBe", "flexibility"),
    (172, "C", "C3", "dfg_motif",          "DFG motif (DFG-in/out kinase conformations)",                "PDB/PDBe", "flexibility"),
    (173, "C", "C3", "back_pocket",        "Back pocket / type II binding site accessible",              "PDB/PDBe", "flexibility"),
    (174, "C", "C3", "selectivity_filter", "Structural selectivity filter residue present",              "PDB/PDBe", "flexibility"),
    (175, "C", "C3", "disordered_loop",    "Disordered loop near binding site",                          "PDB/PDBe", "flexibility"),
    # ---- C4: Binding mode and SAR characteristics (bits 176-191) ----
    (176, "C", "C4", "competitive_inhibition",  "Primary mechanism is competitive inhibition",           "literature", "binding_mode"),
    (177, "C", "C4", "noncompetitive",          "Non-competitive or uncompetitive inhibition",           "literature", "binding_mode"),
    (178, "C", "C4", "fragment_bindable",       "Fragment-sized ligands can bind productively",          "PDB/PDBe",   "binding_mode"),
    (179, "C", "C4", "reversible_binding",      "Binding is reversible under physiological conditions",  "literature", "binding_mode"),
    (180, "C", "C4", "irreversible_mechanism",  "Irreversible / covalent binding mechanism",             "literature", "binding_mode"),
    (181, "C", "C4", "substrate_mimicry",       "Optimal inhibitors mimic the natural substrate",        "literature", "binding_mode"),
    (182, "C", "C4", "transition_state_mimicry","Optimal inhibitors mimic the transition state",         "literature", "binding_mode"),
    (183, "C", "C4", "bisubstrate_mechanism",   "Bisubstrate mechanism (two binding events required)",   "literature", "binding_mode"),
    (184, "C", "C4", "covalent_warhead",        "Covalent warhead compatible with target chemistry",     "PDB/PDBe",   "binding_mode"),
    (185, "C", "C4", "macrocycle_compatible",   "Macrocyclic ligands can access the binding site",       "PDB/PDBe",   "binding_mode"),
    (186, "C", "C4", "peptide_compatible",      "Peptide-derived inhibitors documented",                 "literature", "binding_mode"),
    (187, "C", "C4", "narrow_sar",              "Narrow SAR (small chemical changes have large effect)", "literature", "binding_mode"),
    (188, "C", "C4", "broad_sar",               "Broad SAR (target accommodates diverse chemotypes)",    "literature", "binding_mode"),
    (189, "C", "C4", "selectivity_cliff",       "Selectivity cliff documented (key selectivity residue)","literature", "binding_mode"),
    (190, "C", "C4", "multi_site_binding",      "Multiple independent binding sites documented",         "PDB/PDBe",   "binding_mode"),
    (191, "C", "C4", "cryptic_druggable",       "Cryptic/allosteric site accessible to small molecules", "PDB/PDBe",   "binding_mode"),
]

# ---------------------------------------------------------------------------
# Module D  (bits 192-255)  --  Endogenous Ligand and Pharmacological Context
# ---------------------------------------------------------------------------
# D1: bits 192-215  (24 bits) Endogenous ligand type
# D2: bits 216-231  (16 bits) Native ligand physicochemistry
# D3: bits 232-247  (16 bits) Ligand-receptor kinetics
# D4: bits 248-255  ( 8 bits) Pharmacological context

_MODULE_D = [
    # ---- D1: Endogenous ligand type (bits 192-215) ----
    (192, "D", "D1", "endo_monoamine",    "Endogenous monoamine (dopamine, serotonin, norepinephrine)",  "IUPHAR", "ligand_type"),
    (193, "D", "D1", "endo_amino_acid",   "Endogenous amino acid neurotransmitter (Glu, GABA, Gly)",     "IUPHAR", "ligand_type"),
    (194, "D", "D1", "endo_purine",       "Endogenous purine ligand (ATP, adenosine)",                   "IUPHAR", "ligand_type"),
    (195, "D", "D1", "endo_lipid",        "Endogenous lipid (fatty acid, prostaglandin)",                 "IUPHAR", "ligand_type"),
    (196, "D", "D1", "endo_steroid",      "Endogenous steroid hormone (estrogen, androgen, corticoid)",  "IUPHAR", "ligand_type"),
    (197, "D", "D1", "endo_peptide_small","Small endogenous peptide (<10 AA)",                           "IUPHAR", "ligand_type"),
    (198, "D", "D1", "endo_peptide_large","Large endogenous peptide / protein ligand (>=10 AA)",         "IUPHAR", "ligand_type"),
    (199, "D", "D1", "endo_prostanoid",   "Prostanoid / eicosanoid ligand (PGE2, TXA2, LTB4)",           "IUPHAR", "ligand_type"),
    (200, "D", "D1", "endo_cannabinoid",  "Endocannabinoid ligand (AEA, 2-AG)",                          "IUPHAR", "ligand_type"),
    (201, "D", "D1", "endo_vitamin",      "Vitamin-derived endogenous ligand (retinoic acid, D3)",       "IUPHAR", "ligand_type"),
    (202, "D", "D1", "endo_eicosanoid",   "Eicosanoid / oxylipid ligand",                                "IUPHAR", "ligand_type"),
    (203, "D", "D1", "endo_phospholipid", "Phospholipid endogenous ligand (LPA, S1P)",                   "IUPHAR", "ligand_type"),
    (204, "D", "D1", "endo_ion",          "Ion as primary endogenous activator (Ca2+, H+, Zn2+)",        "IUPHAR", "ligand_type"),
    (205, "D", "D1", "endo_cofactor",     "Enzymatic cofactor (NAD+, FAD, CoA)",                         "IUPHAR", "ligand_type"),
    (206, "D", "D1", "endo_nucleotide",   "Nucleotide ligand (GTP, cAMP, IP3)",                          "IUPHAR", "ligand_type"),
    (207, "D", "D1", "endo_gas",          "Gasotransmitter (NO, CO, H2S)",                               "IUPHAR", "ligand_type"),
    (208, "D", "D1", "endo_glycan",       "Glycan / carbohydrate endogenous ligand",                     "IUPHAR", "ligand_type"),
    (209, "D", "D1", "endo_neuropeptide", "Neuropeptide endogenous ligand",                              "IUPHAR", "ligand_type"),
    (210, "D", "D1", "endo_cytokine",     "Cytokine / growth factor natural ligand",                     "IUPHAR", "ligand_type"),
    (211, "D", "D1", "endo_hormone_amine","Thyroid / catecholamine hormone",                             "IUPHAR", "ligand_type"),
    (212, "D", "D1", "endo_polyamine",    "Polyamine endogenous modulator (spermine, spermidine)",       "IUPHAR", "ligand_type"),
    (213, "D", "D1", "endo_melatonin",    "Melatonin / indoleamine endogenous ligand",                   "IUPHAR", "ligand_type"),
    (214, "D", "D1", "endo_opioid",       "Endogenous opioid peptide (enkephalin, endorphin)",           "IUPHAR", "ligand_type"),
    (215, "D", "D1", "endo_orphan",       "Orphan receptor (endogenous ligand unknown)",                 "IUPHAR", "ligand_type"),
    # ---- D2: Native ligand physicochemistry (bits 216-231) ----
    (216, "D", "D2", "endo_mw_small",     "Small endogenous ligand (MW < 300 Da)",                       "IUPHAR", "endo_physchem"),
    (217, "D", "D2", "endo_mw_medium",    "Medium endogenous ligand (MW 300-600 Da)",                    "IUPHAR", "endo_physchem"),
    (218, "D", "D2", "endo_mw_large",     "Large endogenous ligand (MW > 600 Da)",                       "IUPHAR", "endo_physchem"),
    (219, "D", "D2", "endo_lipophilic",   "Lipophilic endogenous ligand (logP > 2)",                     "IUPHAR", "endo_physchem"),
    (220, "D", "D2", "endo_hydrophilic",  "Hydrophilic endogenous ligand (logP < 0)",                    "IUPHAR", "endo_physchem"),
    (221, "D", "D2", "endo_charged",      "Endogenous ligand bears formal charge at physiological pH",   "IUPHAR", "endo_physchem"),
    (222, "D", "D2", "endo_flexible",     "Highly flexible endogenous ligand (>= 10 rotatable bonds)",   "IUPHAR", "endo_physchem"),
    (223, "D", "D2", "endo_rigid",        "Rigid endogenous ligand (<= 3 rotatable bonds)",              "IUPHAR", "endo_physchem"),
    (224, "D", "D2", "endo_aromatic",     "Aromatic endogenous ligand (contains aromatic ring)",         "IUPHAR", "endo_physchem"),
    (225, "D", "D2", "endo_hbd_donor",    "Endogenous ligand is a significant hydrogen-bond donor",      "IUPHAR", "endo_physchem"),
    (226, "D", "D2", "endo_hba_acceptor", "Endogenous ligand is a significant hydrogen-bond acceptor",   "IUPHAR", "endo_physchem"),
    (227, "D", "D2", "endo_macromolecule","Endogenous ligand is a macromolecule (protein, RNA)",         "IUPHAR", "endo_physchem"),
    (228, "D", "D2", "endo_ro5_compliant","Endogenous ligand is rule-of-five compliant",                 "IUPHAR", "endo_physchem"),
    (229, "D", "D2", "endo_beyond_ro5",   "Endogenous ligand is beyond rule-of-five",                    "IUPHAR", "endo_physchem"),
    (230, "D", "D2", "endo_chiral",       "Endogenous ligand has defined stereocenters",                 "IUPHAR", "endo_physchem"),
    (231, "D", "D2", "endo_symmetric",    "Endogenous ligand has molecular symmetry",                    "IUPHAR", "endo_physchem"),
    # ---- D3: Ligand-receptor kinetics (bits 232-247) ----
    (232, "D", "D3", "kon_fast",          "Fast association rate (kon > 1e6 M-1 s-1)",                   "IUPHAR", "kinetics"),
    (233, "D", "D3", "kon_slow",          "Slow association rate (diffusion-limited or hindered entry)",  "IUPHAR", "kinetics"),
    (234, "D", "D3", "koff_fast",         "Fast dissociation rate (koff > 0.1 s-1)",                     "IUPHAR", "kinetics"),
    (235, "D", "D3", "koff_slow",         "Slow dissociation rate (koff < 0.001 s-1)",                   "IUPHAR", "kinetics"),
    (236, "D", "D3", "affinity_pm",       "Endogenous agonist affinity <= 1 nM (picomolar range)",       "IUPHAR", "kinetics"),
    (237, "D", "D3", "affinity_nm",       "Endogenous agonist affinity 1-100 nM",                        "IUPHAR", "kinetics"),
    (238, "D", "D3", "affinity_um",       "Endogenous agonist affinity > 100 nM",                        "IUPHAR", "kinetics"),
    (239, "D", "D3", "local_conc_high",   "Target operates under high local concentration of ligand",    "IUPHAR", "kinetics"),
    (240, "D", "D3", "systemic_exposure", "Target operates under systemic / circulating ligand levels",  "IUPHAR", "kinetics"),
    (241, "D", "D3", "pulsatile_release", "Endogenous ligand released in pulsatile fashion",             "IUPHAR", "kinetics"),
    (242, "D", "D3", "tonic_release",     "Endogenous ligand tonically present",                         "IUPHAR", "kinetics"),
    (243, "D", "D3", "receptor_reserve",  "Spare receptor / receptor reserve documented",               "IUPHAR", "kinetics"),
    (244, "D", "D3", "partial_agonism",   "Endogenous ligand is a partial agonist at this target",       "IUPHAR", "kinetics"),
    (245, "D", "D3", "inverse_agonism",   "Endogenous mechanism can be modulated by inverse agonists",   "IUPHAR", "kinetics"),
    (246, "D", "D3", "rapid_recycling",   "Endogenous ligand rapidly recycled / inactivated",            "IUPHAR", "kinetics"),
    (247, "D", "D3", "slow_clearance",    "Endogenous ligand slowly cleared (long half-life)",           "IUPHAR", "kinetics"),
    # ---- D4: Pharmacological context (bits 248-255) ----
    (248, "D", "D4", "approved_drugs",    "Target has at least one approved drug",                       "ChEMBL", "pharma_context"),
    (249, "D", "D4", "clinical_trials",   "Target has compounds in clinical trials",                     "ChEMBL", "pharma_context"),
    (250, "D", "D4", "tool_compounds",    "Validated chemical probe / tool compound available",           "ChEMBL", "pharma_context"),
    (251, "D", "D4", "genetic_validation","Target genetically validated (knockout, GWAS, Mendelian)",    "ChEMBL", "pharma_context"),
    (252, "D", "D4", "druggable_genome",  "Target is part of the druggable genome list",                 "ChEMBL", "pharma_context"),
    (253, "D", "D4", "polypharmacology_common", "Target frequently co-targeted (known polypharmacology)","ChEMBL", "pharma_context"),
    (254, "D", "D4", "resistance_mutations","Resistance mutations documented for this target",           "ChEMBL", "pharma_context"),
    (255, "D", "D4", "biomarker_available","Validated biomarker available for target engagement",        "ChEMBL", "pharma_context"),
]

# ---------------------------------------------------------------------------
# Build the full CRAFT_SCHEMA list
# ---------------------------------------------------------------------------

CRAFT_SCHEMA: list[BitDefinition] = []

for raw_list in (_MODULE_A, _MODULE_B, _MODULE_C, _MODULE_D):
    for row in raw_list:
        pos, mod, sb, name, desc, src, cat = row
        CRAFT_SCHEMA.append(BitDefinition(
            position=pos,
            module=mod,
            sub_block=sb,
            name=name,
            description=desc,
            source=src,
            category=cat,
        ))

assert len(CRAFT_SCHEMA) == BIT_COUNT, f"Schema has {len(CRAFT_SCHEMA)} entries, expected {BIT_COUNT}"

# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

SCHEMA_BY_POSITION: dict[int, BitDefinition] = {b.position: b for b in CRAFT_SCHEMA}
SCHEMA_BY_NAME: dict[str, BitDefinition] = {b.name: b for b in CRAFT_SCHEMA}

MODULE_RANGES = {
    "A": range(0, 64),
    "B": range(64, 128),
    "C": range(128, 192),
    "D": range(192, 256),
}

# Default class-based pocket profiles for Module C (Tier 2 fallback)
# Used when no PDB structure is available for a target.
# Bit positions within Module C (absolute positions 128-191).
CLASS_BASED_POCKET_DEFAULTS: dict[str, list[int]] = {
    "GPCR":            [128, 131, 133, 136, 144, 149, 164, 176, 179, 187],
    "Kinase":          [128, 130, 132, 138, 141, 144, 146, 160, 168, 169, 171, 172, 176, 179],
    "Ion Channel":     [130, 136, 145, 150, 152, 160, 176, 179],
    "Nuclear Receptor":[128, 130, 132, 136, 144, 160, 164, 176, 179, 187],
    "Transporter":     [128, 131, 133, 136, 146, 160, 176, 179],
    "Enzyme":          [128, 130, 133, 138, 144, 146, 160, 164, 176, 179],
    "Protease":        [128, 130, 132, 136, 138, 144, 150, 152, 160, 176, 179],
    "Phosphatase":     [128, 130, 138, 146, 152, 160, 176, 179],
    "E3 Ligase":       [128, 131, 135, 140, 146, 162, 165, 176, 179],
    "Epigenetic":      [128, 131, 133, 136, 146, 152, 162, 165, 176, 179],
    "Transcription Factor": [128, 131, 135, 140, 146, 162, 167, 176, 179],
}
