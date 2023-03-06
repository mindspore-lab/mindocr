from typing import List
import mindspore as ms
from addict import Dict
from .det_dataset import DetDataset
from .rec_dataset import RecDataset
from .rec_lmdb_dataset import LMDBDataset

supported_dataset_types = ['BaseDataset', 'DetDataset', 'RecDataset', 'LMDBDataset']

def build_dataset(
        dataset_config: dict,
        loader_config: dict,
        num_shards=None,
        shard_id=None,
        is_train=True,
        **kwargs,
        ):
    '''
    Args:
        dataset_config (dict): dataset reading and processing configuartion containing keys:
            - type: dataset type, 'DetDataset', 'RecDataset'
            - data_dir: folder to the dataset. 
            - annot_file (optional for recognition): absolute file path to the annotation file
            - transform_pipeline (list[dict]): config dict for image and label transformation
        loader_config (dict): dataloader configuration containing keys:
            - batch_size: batch size for data loader 
            - drop_remainder: whether to drop the data in the last batch when the total of data can not be divided by the batch_size
        num_shards: num of devices for distributed mode
        shard_id: device id 
        is_train: whether it is in training stage 

    Return:
        data_loader (Dataset): dataloader to generate data batch 
    '''
    # build datasets
    dataset_class_name = dataset_config.pop('type')
    assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)

    #print('dataset config', dataset_config)
    
    dataset_args = dict(is_train=is_train, **dataset_config) 
    dataset = dataset_class(**dataset_args)

    # create batch loader
    dataset_column_names = dataset.get_column_names()
    print('==> Dataset columns: \n\t', dataset_column_names)
    
    # TODO: the optimal value for prefetch. * num_workers?
    #ms.dataset.config.set_prefetch_size(int(loader_config['batch_size']))
    #print('prfectch size:', ms.dataset.config.get_prefetch_size())

    # TODO: config multiprocess and shared memory
    ds = ms.dataset.GeneratorDataset(
                    dataset,
                    column_names=dataset_column_names,
                    num_parallel_workers=loader_config['num_workers'],
                    num_shards=num_shards,
                    shard_id=shard_id,
                    python_multiprocessing=True,
                    max_rowsize =loader_config['max_rowsize'], 
                    shuffle=loader_config['shuffle'],
                    )

    # TODO: set default value for drop_remainder and max_rowsize
    dataloader = ds.batch(loader_config['batch_size'],
                    drop_remainder=loader_config['drop_remainder'],
                    max_rowsize=loader_config['max_rowsize'],
                    #num_parallel_workers=loader_config['num_workers'],
                    )

    #steps_pre_epoch = dataset.get_dataset_size()
    return dataloader