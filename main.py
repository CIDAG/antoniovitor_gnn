from pathlib import Path
import sys
from datasource.pcqm4mv2.PCQMDataset import PCQMDataset
import seed


datasets_root = Path('tests/')

def main(params):
    # Seeds all libraries
    seed.run()

    # Parameters
    dataset = params['dataset']
    name = params['name']
    target_property = int(params['target_property'])

    # temp
    dataset_path = datasets_root / dataset / name
    dataset = PCQMDataset(dataset_path)


    # dataset = datasets.get_dataset(dataset_name, dataset_version, target_property)
    # print(dataset)
    




def get_params():
    action = sys.argv[1]
    params = dict([param.split('=') for param in sys.argv[2:]])
    params['action'] = action
    return params

if __name__ == '__main__':
    params = get_params()
    main(params)