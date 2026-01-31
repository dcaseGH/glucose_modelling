from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, dft

# 1. Generate 3D Geometry of Glucose using RDKit
# SMILES for alpha-D-glucopyranose
smiles = "C1[C@@H]([C@H]([C@@H]([C@H](O1)O)O)O)O"
mol = Chem.MolFromSmiles(smiles)

# Add Hydrogens (important for quantum chem)
mol = Chem.AddHs(mol)

# Generate initial 3D coordinates (ETKDG method)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

# Quick optimization with MMFF94 force field to "clean up" bonds
AllChem.MMFFOptimizeMolecule(mol)

# 2. Extract coordinates for PySCF
# We convert the RDKit molecule into an XYZ string format
xyz_data = ""
conf = mol.GetConformer()
for i, atom in enumerate(mol.GetAtoms()):
    pos = conf.GetAtomPosition(i)
    xyz_data += f"{atom.GetSymbol()} {pos.x} {pos.y} {pos.z}\n"

# 3. Calculate Single Point Energy with PySCF
# Define the molecular system
pyscf_mol = gto.Mole()
pyscf_mol.atom = xyz_data
pyscf_mol.basis = 'sto-3g'  # Using a small basis set for speed; use '6-31G*' for better accuracy
pyscf_mol.build()

# Define the method (Density Functional Theory - B3LYP)
mf = dft.RKS(pyscf_mol)
mf.xc = 'b3lyp'

print("Starting Single Point Energy Calculation...")
energy = mf.kernel()

print(f"\nTotal Energy (Hartree): {energy}")