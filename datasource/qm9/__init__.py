from pathlib import Path
import torch
from torch_geometric import transforms
from datasource.qm9.QM9Dataset import QM9Dataset
from datasource.qm9.transformations import Complete, SelectTargetProperty

class QM9():
    def get_dataset(self, version, target):
        transform = transforms.Compose([
            SelectTargetProperty(target),
            Complete(),
            transforms.Distance(norm=False)]
        )

        dataset_path = Path(f'datasets/qm9/{version}')
        dataset = QM9Dataset(dataset_path, transform=transform).shuffle()

        return dataset
