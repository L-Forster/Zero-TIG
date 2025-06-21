from .multi_read_data import *

def CreateDataset(args, task):
    dataset = None
    dataset_name = args.dataset

    if dataset_name in ['lowlight_dataset', 'RLV', 'BVI-RLV']:
        dataset = RLVDataLoader()
    elif dataset_name in ['DID', 'DID_1080']:
        dataset = DidDataloader()
    elif dataset_name in ['SDSD', '3_SDSD']:
        dataset = SDSDDataloader()
    else:
        dataset = DefaultDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(args, task)
    return dataset