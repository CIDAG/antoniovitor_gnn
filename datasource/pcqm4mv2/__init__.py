from pathlib import Path
import torch
from torch_geometric import transforms
from datasource.pcqm4mv2.PCQMDataset import PCQMDataset

class PCQM4Mv2():
    def get_dataset(self, name, target):
        dataset_path = Path(f'datasets/pcqm4mv2/{target}')
        dataset = PCQMDataset(dataset_path).shuffle()

        return dataset
