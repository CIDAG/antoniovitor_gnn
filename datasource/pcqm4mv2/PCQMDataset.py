import math
import shutil
import torch
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
    extract_zip,
    extract_tar,
    extract_gz,
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

def get_mol_idx(mol):
    name = mol.GetProp("_Name")
    idx = name[name.rfind('/')+1:-4]
    return int(idx)

################################################################################
# Main
################################################################################
class PCQMDataset(InMemoryDataset):
    properties_filename = Path('data.csv')
    properties_url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'
    
    geometries_filename = Path('pcqm4m-v2-train.sdf')
    geometries_url = 'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2-train.sdf.tar.gz'

    _idx_list = None

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def idx_list(self):
        if self._idx_list is None:
            source = Path(self.processed_dir) / 'idx_list.csv'
            self._idx_list = pd.read_csv(source)
        return self._idx_list

    @property
    def raw_file_names(self):
        return [
            self.properties_filename,
            self.geometries_filename,
        ]

    @property
    def processed_file_names(self):
        idx_list = self.idx_list
        filenames = [f'data.{i}.pt' for i, idx in enumerate(idx_list)]
        return filenames

    def len(self):
        return len(self.idx_list)

    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / f'data.{idx}.pt')
        return data

    def download(self):
        # Download properties file
        if not os.path.exists(self.raw_dir / self.properties_filename):
            print(f'Downloading properties file...')
            path = download_url(self.properties_url, self.raw_dir, log=False)
            extract_zip(path, self.raw_dir, log=False)
            os.unlink(path)

            # Move files
            origin = Path(self.raw_dir) / 'pcqm4m-v2/raw/data.csv.gz'
            destiny = Path(self.raw_dir) / 'data.csv.gz'
            os.rename(origin, destiny)
            shutil.rmtree(Path(self.raw_dir) / 'pcqm4m-v2')

            # Extract data
            origin = Path(self.raw_dir) / 'data.csv.gz'
            destiny = Path(self.raw_dir) / 'data.csv'
            extract_gz(origin, destiny, log=False)
            os.unlink(origin)

        # # Download geometries file
        if not os.path.exists(self.raw_dir / self.geometries_filename):
            print(f'Downloading geometries file...')
            path = download_url(self.geometries_url, self.raw_dir)
            if(generate_file_md5(path) != 'fd72bce606e7ddf36c2a832badeec6ab'):
                raise ValueError('MD5 Hash does not match the original file')
            extract_tar(path, self.raw_dir, log=False)
            os.unlink(path)

    def process(self):
        processed_path = Path(self.processed_dir)
        prop_data = pd.read_csv(Path(self.raw_dir) / 'data.csv').to_numpy()
        supplier = Chem.SDMolSupplier(
            str(Path(self.raw_dir) / 'pcqm4m-v2-train.sdf'),
            removeHs=False,
            sanitize=False)

        idx_list = []
        for i, mol in enumerate(tqdm(supplier)):
            idx = get_mol_idx(mol)
            idx_list.append(idx)
            # homolumogap is at column 2
            property = prop_data[idx][2]
            if not math.isnan(property):
                torch_data = process_molecule(idx, mol, property)
                torch.save(torch_data, processed_path / (f'data.{i}.pt'))

        df = pd.DataFrame(idx_list, columns=['idx'])
        df.to_csv(Path(self.processed_dir) / 'idx_list.csv')
