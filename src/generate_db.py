import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
#from copy import deepcopy
import h5py
from pyscf import gto, dft

n_displacements = 100  # randomized displacements per conformer

def save_to_h5(filename, mol_name, conf_idx, coords, energy, method_info):
    """
    Saves molecular data to an HDF5 file.
    """
    with h5py.File(filename, "a") as f:
        # Create a group for the molecule type (e.g., 'glucose')
        grp = f.require_group(mol_name)
        
        # Create a unique name for this specific displacement/conformer
        dset_name = f"conf_{conf_idx:04d}"
        
        # If the dataset exists, delete it so we can overwrite/update
        if dset_name in grp:
            del grp[dset_name]
            
        dset = grp.create_dataset(dset_name, data=coords)
        
        # Store metadata as attributes
        dset.attrs["energy"] = energy
        #dset.attrs["basis"] = method_info['basis']
        #dset.attrs["xc"] = method_info['xc']
        dset.attrs["unit"] = "Hartree"

def create_displaced_mol(original_mol, stdev=0.1, confId=-1):
    # 1. Create a deep copy so the original stays safe
    new_mol = Chem.Mol(original_mol)
    
    # 2. Get the conformer and current positions
    conf = new_mol.GetConformer()
    pos = conf.GetPositions()
    
    # 3. Generate and apply random noise
    # Standard deviation of 0.1A is a "gentle shake"
    noise = np.random.normal(0, stdev, size=pos.shape)
    new_pos = pos + noise
    
    # 4. Update the coordinates in the new object
    for i in range(new_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, new_pos[i])
    
    return new_mol

# xtb isn't installing as dont need dft at this stage
#method_info = {'basis': 'sto-3g', 'xc': 'pbe'}
method_info = {'method': 'forcefield', 'forcefield': 'MMFF94'}
smiles = "C1[C@@H]([C@H]([C@@H]([C@H](O1)O)O)O)O"
mol = Chem.MolFromSmiles(smiles)

# Add Hydrogens (important for quantum chem)
mol = Chem.AddHs(mol)

# Generate initial 3D coordinates (ETKDG method)
AllChem.EmbedMolecule(mol, AllChem.ETKDG())

# Quick optimization with MMFF94 force field to "clean up" bonds
AllChem.MMFFOptimizeMolecule(mol)
# properties of the molecule for ff calcn
mp = AllChem.MMFFGetMoleculeProperties(mol)

# Generate some conformers
conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=50, pruneRmsThresh=0.5)
print(f"Generated {len(conf_ids)} conformers.")
print(f"Will create {n_displacements} displacements per conformer.")

for conf_id in conf_ids:
    # Make variations thereupon
    for i in range(n_displacements):
        #print(f"Processing iteration {i}...")
        
        # 1. Create a newly displaced molecule
        temp_mol = create_displaced_mol(mol, stdev=0.05, confId=conf_id) 
    
        # 2. Extract coordinates for PySCF
        #conf = temp_mol.GetConformer()
        atoms = [atom.GetSymbol() for atom in temp_mol.GetAtoms()]
        positions = temp_mol.GetConformer().GetPositions()
    
        # Format for PySCF: "Atom X Y Z; Atom X Y Z"
        #atom_str = "; ".join([f"{atoms[j]} {positions[j][0]} {positions[j][1]} {positions[j][2]}" 
        #                     for j in range(len(atoms))])
    
        # 3. Run PySCF Calculation
        #pyscf_mol = gto.M(atom=atom_str, basis=method_info['basis'])
        #mf = dft.RKS(pyscf_mol)
        #mf.xc = method_info['xc']
        #energy = mf.kernel()

        # get energy from rdkit for now
        #energy = AllChem.MMFFGetMoleculeForceField(temp_mol).CalcEnergy()
        energy = AllChem.MMFFGetMoleculeForceField(temp_mol, mp).CalcEnergy()

        # 4. Save to Database
        save_to_h5('glucose_data.h5', "glucose_alpha", i, positions, energy, method_info)

print("\nAll calculations saved to 'glucose_data.h5'")

# Quick check: Read back the energy of the 3rd conformer
with h5py.File("glucose_data.h5", "r") as f:
    energy_val = f["glucose_alpha/conf_0002"].attrs["energy"]
    coords = f["glucose_alpha/conf_0002"][:]
    
    print(f"Retrieved Energy: {energy_val}")
    print(f"Coordinates Shape: {coords.shape}")