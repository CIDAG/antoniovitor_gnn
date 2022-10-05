from pathlib import Path
import sys
from monitoring.log import FileLog

from s2s_gnn import e2eGNN
from datasource.pcqm4mv2.PCQMDataset import PCQMDataset
import seed


datasets_root = Path('tests/')

def main(params):
    # Seeds all libraries
    seed.run()

    # Parameters
    dataset_name = params['dataset']
    name = params['name']
    target_property = int(params['target_property'])

    # dataset
    dataset_path = datasets_root / dataset_name / name
    dataset = PCQMDataset(dataset_path).shuffle()

    # directory for saving training
    model_dir = Path(f'/saved_models/dataset-{dataset_name}-name={name}-property-{target_property}')

    # log
    log = FileLog(model_dir / 'logs')

    model = e2eGNN(dataset, model_dir, log)
    model.train()

def get_params():
    action = sys.argv[1]
    params = dict([param.split('=') for param in sys.argv[2:]])
    params['action'] = action
    return params

if __name__ == '__main__':
    params = get_params()
    main(params)