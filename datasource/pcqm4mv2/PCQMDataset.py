import shutil
import time
import torch
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
    extract_zip,
    extract_tar
)
from rdkit import Chem
from openbabel import pybel
import numpy as np
import torch
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
import pandas as pd
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_scatter import scatter
from torch_geometric.data import Data
from pathlib import Path
from torch_geometric.utils import remove_self_loops
from tqdm import tqdm
import os
from trie import Trie

from utils import generate_file_md5

################################################################################
# constants
################################################################################

types = {
    'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4,
    'He': 5, 'Cl':6, 'S': 7, 'Si': 8, 'Ca': 9,
    'P': 10, 'Be': 11, 'Zn': 12, 'As': 13, 'Ar': 14,
    'B': 15, 'Se': 16, 'Mg': 17, 'Ti': 18, 'Br': 19,
    'Ge': 20, 'Ga': 21,
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

################################################################################
# Transforms
################################################################################
def complete_transform(data):
    # Pegando o dispositivo para salvar os novos tensores
    # CPU ou GPU
    device = data.edge_index.device

    # Criando novos row e col (cada tensor terá dimensão igual a quantidade de átomos na molécula)
    row = torch.arange(data.num_nodes, dtype=torch.long, device=device) # torch.arange = np.arange
    # neste caso, torch.arange irá criar um tensor com valores começando no 0 
    # e terminando em data.num_nodes com step = 1
    col = torch.arange(data.num_nodes, dtype=torch.long, device=device) 

    # Basicamente, aqui eu estou repetindo cada elemento do row x vezes
    # onde x é igual a quantidade de átomos na molécula (data.num_nodes)
    # por exemplo: se inicialmente row = [0,1,2] e a molécula contém 3 átomos
    # após essa operações, row vai ser igual a [0,0,0,1,1,1,2,2,2]
    row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
    # mesma ideia com as colunas ... porém aqui os idx estão intercalados
    # por exemplo, [0,1,2,0,1,2,0,1,2]
    col = col.repeat(data.num_nodes)
    # Com isso, edge_index contém todas as possíveis conexões (é a matriz do grafo completo)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = None
    # Se existir atributos para as arestas, faça
    if data.edge_attr is not None:
        
        idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
        # basicamente, pegando o .shape do tensor
        size = list(data.edge_attr.size())
        # modificando a quantidade de linhas do tensor
        # a ideia aqui é criar um novo size para representar o edge_attr do grafo completo
        size[0] = data.num_nodes * data.num_nodes
        
        # Criando o novo edge_attr do grafo completo
        # note que inicialmente os atributos das arestas estão todos zerados (new_zeros) 
        edge_attr = data.edge_attr.new_zeros(size)
        # com base no index recuperado no inicio do if, irei atualizar os atributos
        # das arestas que realmente existem no grafo
        # as demais arestas ficaram zeradas, ou seja, não existem ...
        edge_attr[idx] = data.edge_attr

    # como edge_index e edge_attr representam um grafo completo (incluindo arestas laços)
    # a função remove_self_loops irá remover esses laços
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    data.edge_attr = edge_attr
    data.edge_index = edge_index

    return data

################################################################################
# Utils
################################################################################
def get_atoms_positions(molecule):
    conformer = molecule.GetConformer()

    positions = []
    for index in range(molecule.GetNumAtoms()):
        pos = conformer.GetAtomPosition(index)
        pos = [pos.x, pos.y, pos.z]
        positions.append(pos)
    
    return positions

def prepare_canonical_smiles(raw_dir):
    df = pd.read_csv(Path(raw_dir) / 'data.csv.gz')
    df = df.dropna(subset='homolumogap')

    for i, row in tqdm(df.iterrows(), total=len(df.index)):
        smiles = df.at[i, 'smiles']
        df.at[i, 'smiles'] = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)

    df.to_csv(Path(raw_dir) / 'data-with-canonical-smiles.csv')

def find_molecule_data(mol, df):
    mol_without_hs = Chem.RemoveHs(mol)
    canon_smiles = Chem.MolToSmiles(mol_without_hs, canonical=True)
    row = df.loc[df['smiles'] == canon_smiles]

    print(canon_smiles)
    print(row)
    return row.to_dict()

def process_molecule(molecule, target_property):
    y = target_property
    positions = get_atoms_positions(molecule)
    positions = torch.tensor(positions, dtype=torch.float)

    N = molecule.GetNumAtoms()

    type_idx = []
    atomic_number = []
    aromatic = []
    sp = []
    sp2 = []
    sp3 = []
    num_hs = []
    for atom in molecule.GetAtoms():
        type_idx.append(types[atom.GetSymbol()]) # Átomo
        atomic_number.append(atom.GetAtomicNum()) # Número atomico
        aromatic.append(1 if atom.GetIsAromatic() else 0) # Aromaticidade
        hybridization = atom.GetHybridization() # Hibridização
        sp.append(1 if hybridization == HybridizationType.SP else 0) # SP
        sp2.append(1 if hybridization == HybridizationType.SP2 else 0) # SP2
        sp3.append(1 if hybridization == HybridizationType.SP3 else 0) # SP3

    z = torch.tensor(atomic_number, dtype=torch.long)

    row, col, edge_type = [], [], []
    for i, bond in enumerate(molecule.GetBonds()):
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]


    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    # Criando feature vector final para cada átomo
    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                        dtype=torch.float).t().contiguous()
    x = torch.cat([x1.to(torch.float), x2], dim=-1)

    name = molecule.GetProp('_Name')

    data = Data(x=x, z=z, pos=positions, edge_index=edge_index,
                edge_attr=edge_attr, y=y, name=name, idx=i)

    data = complete_transform(data)
    data = T.Distance(norm=False)(data)

    return data


################################################################################
# Main
################################################################################
def process_dataset(raw_dir):
    # check if property data was processed
    if not os.path.exists(Path(raw_dir) / 'data-with-canonical-smiles.csv'):
        prepare_canonical_smiles(raw_dir)
    
    # property dataframe
    df = pd.read_csv(Path(raw_dir) / 'data-with-canonical-smiles.csv')
    df.dropna()

    # molecular geometry supplier
    supplier = Chem.SDMolSupplier(str(Path(raw_dir) / 'pcqm4m-v2-train.sdf'), removeHs=False, sanitize=False)

    # process dataset
    dataset = []
    for mol in supplier:
        target_property = find_molecule_data(mol, df)
        
        # skips if target property doesn't exists
        if target_property is None: continue

        data = process_molecule(mol, target_property)
        dataset.append(data)

    return dataset


class PCQMDataset(InMemoryDataset):
    url = [
        'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip',
        'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz',
    ]

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv.gz', 'pcqm4m-v2-train.sdf']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # First download
        path = download_url(self.url[0], self.raw_dir)
        extract_zip(path, self.root)
        os.unlink(path)

        # Move files
        origin = Path(self.root) / 'pcqm4m-v2/raw/data.csv.gz'
        destiny = Path(self.raw_dir) / 'data.csv.gz'
        os.rename(origin, destiny)
        shutil.rmtree(Path(self.root) / 'pcqm4m-v2')

        # Second download
        path = download_url(self.url[1], self.raw_dir)
        if(generate_file_md5(path) != 'fd72bce606e7ddf36c2a832badeec6ab'):
            raise ValueError('MD5 Hash does not match the original file')
        extract_tar(path, self.raw_dir)
        os.unlink(path)

    def process(self):
        # Prepare file with valid molecules (with DFT geometry and property defined)
        self.prepare_valid_molecules()


        # dataset = process_dataset(self.raw_dir)
        # torch.save(self.collate(dataset), self.processed_paths[0])
    
    def get_molecules_with_canonical_smiles(self):
        source_path = Path(self.raw_dir) / 'data.csv.gz'
        destination_path = Path(self.raw_dir) / 'data-with-canonical-smiles.csv'

        # Returns cached fle if exists
        if os.path.exists(destination_path):
            return pd.read_csv(destination_path)

        df = pd.read_csv(source_path)
        df = df.dropna()

        for i, row in tqdm(df.iterrows(), total=len(df.index)):
            mol = Chem.MolFromSmiles(row['smiles'])
            df.at[i, 'smiles'] = Chem.MolToSmiles(mol, canonical=True)

        df.dropna()

        # Caches results
        df.to_csv(destination_path)

        return df
    
    def get_valid_molecules(self, df):
        source_path = str(Path(self.raw_dir) / 'pcqm4m-v2-train.sdf')
        destination_path = Path(self.raw_dir) / 'valid_mols.csv'
        
        # Returns cached fle if exists
        if os.path.exists(destination_path):
            return pd.read_csv(destination_path)

        print('Creating trie structure...')
        data = []
        for i, row in tqdm(df.iterrows(), total=len(df.index)):
            item = [row['smiles'], (row['idx'],row['homolumogap'])]
            data.append(item)
        trie = Trie(data)

        supplier = Chem.SDMolSupplier(source_path, removeHs=False, sanitize=False)

        # Filter valid molecules
        print('Filtering valid molecules...')
        valid_mols = []
        for mol in tqdm(supplier):
            mol_without_hs = Chem.RemoveHs(mol)
            canonical_smiles = Chem.MolToSmiles(mol_without_hs, canonical=True)
            mol_data = trie.get(canonical_smiles)
            if mol_data is not None:
                idx, homolumogap = mol_data
                valid_mols.append([idx, canonical_smiles, homolumogap])
        
        # Creating new dataframe
        valid_df = pd.DataFrame(valid_mols, columns=['idx', 'smiles', 'homolumogap'])
        valid_df['idx'] = range(len(valid_df))
        valid_df.set_index('idx', inplace=True)

        # Caches results
        valid_df.to_csv(destination_path)

        return valid_df

    def create_data_files(self, df):
        source_path = str(Path(self.raw_dir) / 'pcqm4m-v2-train.sdf')
        processed_path = Path(self.processed_dir)
        supplier = Chem.SDMolSupplier(source_path, removeHs=False, sanitize=False)

        print('Creating data files...')
        for i, mol in enumerate(tqdm(supplier)):
            id = 0
            if(i < id): continue
            if(i > id): break

            data = find_molecule_data(mol, df)
            torch_data = process_molecule(mol, data['homolumogap'])
            idx = data['idx']
            torch.save(torch_data, processed_path / (f'data.{idx}.pt'))



    def prepare_valid_molecules(self):
        # Generates canonical smiles for molecules
        df_canonical_mol = self.get_molecules_with_canonical_smiles()

        # Filter molecules with dft geometry
        df = self.get_valid_molecules(df_canonical_mol)

        # Creates SDF files for each molecules
        self.create_data_files(df)