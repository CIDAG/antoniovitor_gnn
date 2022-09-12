from pathlib import Path
from rdkit import Chem
from tqdm import tqdm


source_path = 'tests/pcqm/pcqm-dft/raw/pcqm4m-v2-train.sdf'
supplier = Chem.SDMolSupplier(source_path, removeHs=False, sanitize=False)

def get_mol_idx(mol):
    name = mol.GetProp("_Name")
    name = name[name.rfind('/')+1:-4]
    return name

with open('names', 'w') as file:
    for mol in tqdm(supplier):
        idx = get_mol_idx(mol)
        file.write(f'{idx}\n')
