from pathlib import Path
import torch

from datasource.qm9 import QM9
from datasource.pcqm4mv2 import PCQM4Mv2

def get_dataset(name, version, target_property):
    if(name == 'qm9'):
        return QM9().get_dataset(version, target_property)
    
    if(name == 'pcqm'):
        return PCQM4Mv2().get_dataset(version, target_property)
