import pymupdf
import os

def create_sample_study_pdf(output_path="data/sample_biology_notes.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = pymupdf.open()

    # --- Page 1 ---
    page1 = doc.new_page()
    text_p1 = """CHAPTER 1: CELL BIOLOGY & ORGANELLES

1.1 Introduction to the Eukaryotic Cell
Eukaryotic cells are characterized by membrane-bound organelles and a true nucleus enclosed by a nuclear envelope with nuclear pores. 

1.2 Mitochondria: The Powerhouse of the Cell
Mitochondria are double-membraned organelles responsible for generating adenosine triphosphate (ATP) through cellular respiration and oxidative phosphorylation. 
Key features:
- Inner membrane is folded into cristae to dramatically increase surface area for ATP synthase complexes.
- The mitochondrial matrix contains enzymes of the Krebs cycle (Citric Acid Cycle).
- Mitochondria possess their own circular DNA (mtDNA) and 70S ribosomes, supporting the endosymbiotic theory that they originated from engulfed aerobic prokaryotes.

1.3 Endoplasmic Reticulum (ER) and Golgi Complex
- Rough Endoplasmic Reticulum (RER): Studded with ribosomes; synthesizes and processes proteins destined for secretion or lysosomal incorporation.
- Smooth Endoplasmic Reticulum (SER): Lacks ribosomes; synthesizes lipids and steroids, metabolizes carbohydrates, and detoxifies drugs and poisons in liver cells.
- Golgi Apparatus: Consists of flattened cisternae with cis (receiving) and trans (shipping) faces. It modifies glycoproteins, packages enzymes, and tags proteins for transport.
"""
    rect1 = pymupdf.Rect(50, 50, 550, 750)
    page1.insert_textbox(rect1, text_p1, fontsize=11, fontname="helv")

    # --- Page 2 ---
    page2 = doc.new_page()
    text_p2 = """CHAPTER 2: PHOTOSYNTHESIS & BIOENERGETICS

2.1 Chloroplast Structure
Chloroplasts are specialized photosynthetic organelles containing chlorophyll pigments.
- Thylakoids: Flattened disc-like membranes arranged in stacks called grana. Thylakoid membranes harbor Photosystem I (PSI - P700) and Photosystem II (PSII - P680).
- Stroma: The dense fluid surrounding the grana where carbon fixation enzymes operate.

2.2 Light-Dependent Reactions
Occurs on thylakoid membranes:
1. Photolysis of water: Light energy absorbed by PSII splits water into 2H+, 1/2 O2, and electrons. Oxygen is released as a byproduct.
2. Photophosphorylation: Electron transport generates a proton motive gradient across the thylakoid membrane into the lumen, driving ATP synthesis.
3. High-energy electron carriers NADPH and ATP are exported to the stroma.

2.3 The Calvin Cycle (Light-Independent Reactions)
Occurs in the stroma:
1. Carbon Fixation: Carbon dioxide (CO2) is fixed onto Ribulose 1,5-bisphosphate (RuBP) catalyzed by the enzyme RuBisCO.
2. Reduction: 3-phosphoglycerate (3-PGA) is phosphorylated and reduced by ATP and NADPH to glyceraldehyde 3-phosphate (G3P).
3. Regeneration: For every 3 CO2 molecules fixed, 6 G3P are produced, of which 1 net G3P exits to form glucose, and 5 G3P regenerate RuBP.
"""
    rect2 = pymupdf.Rect(50, 50, 550, 750)
    page2.insert_textbox(rect2, text_p2, fontsize=11, fontname="helv")

    # --- Page 3 ---
    page3 = doc.new_page()
    text_p3 = """CHAPTER 3: CELL DIVISION & DNA REPLICATION

3.1 Mitosis vs. Meiosis
- Mitosis: Equational division occurring in somatic cells, resulting in two genetically identical diploid daughter cells for growth and tissue repair.
  Stages: Prophase (chromatin condenses), Metaphase (chromosomes align on metaphase plate), Anaphase (sister chromatids separate to opposite poles), Telophase and Cytokinesis.
- Meiosis: Reductional division occurring in germ cells, producing four genetically diverse haploid gametes.
  Crossing Over: Homologous chromosomes pair up during Pachynene of Prophase I, exchanging non-sister chromatid segments to generate genetic recombination.

3.2 DNA Replication Mechanisms
DNA replication is semi-conservative, as demonstrated by the classic 1958 Meselson and Stahl experiment using Nitrogen-15 isotope labeling.
Key replication enzymes:
1. DNA Helicase: Unwinds the double helix at the replication fork by breaking hydrogen bonds between base pairs.
2. Single-Strand Binding Proteins (SSBPs): Stabilize unwound single-stranded template DNA.
3. DNA Topoisomerase (Gyrase): Relieves supercoiling tension upstream of the replication fork.
4. RNA Primase: Synthesizes short RNA primers required for DNA polymerase initiation.
5. DNA Polymerase III: Synthesizes new DNA in the 5' to 3' direction. Continuous on the leading strand and discontinuous on the lagging strand (producing Okazaki fragments).
6. DNA Ligase: Seals phosphodiester nicks between Okazaki fragments on the lagging strand.
"""
    rect3 = pymupdf.Rect(50, 50, 550, 750)
    page3.insert_textbox(rect3, text_p3, fontsize=11, fontname="helv")

    doc.save(output_path)
    doc.close()
    print(f"Sample study notes PDF created successfully at: {output_path}")

if __name__ == "__main__":
    create_sample_study_pdf()
