from typing import List
import mindspore as ms
from addict import Dict
from .det_dataset import DetDataset
#from transforms.transforms_factory import create_transforms
#from .rec_dataset import RecLMDBDataset, RecTextDataset

support_dataset_types = ['DetDataset', 'RecDataset']

def build_dataset(dataset_config: dict,
                    loader_config: dict,
                    num_shards=None,
                    shard_id=None,
                    is_train=True,
                    **kwargs):
    '''
    Args:
        dataset_config (dict): config dict for the dataset, required keys:
            type: dataset type, 'DetDataset', 'RecDataset'
            image_dir: directory of data,
                image folder path for detection task
                lmdb folder path for lmdb dataset
                co
            annot_file (optional for recognition): annotation file path
            transform_pipeline (list[dict]): config dict for image and label transformation

    Return:
        data loader, for feeding data batch to network
    '''
    # build datasets
    dataset_class_name = dataset_config.pop('type')
    assert dataset_class_name in support_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)

    print(dataset_config)

    dataset = dataset_class(**dataset_config, is_train=is_train)
    #dataset = dataset_class(dataset_config['data_dir'], dataset_config['label_files'], dataset_config['sample_ratios'], dataset_config['shuffle'], dataset_config['transforms'], is_train=is_train, exclude_output_columns=dataset_config['exclude_output_columns'])

    # create batch loader
    dataset_column_names = dataset.get_column_names()
    print('dataset columns: ', dataset_column_names)
    ds = ms.dataset.GeneratorDataset(dataset,
                        column_names=dataset_column_names,
			num_parallel_workers=loader_config['num_workers'],
                        num_shards=num_shards,
                        shard_id=shard_id,
                        shuffle=loader_config['shuffle'])

    # TODO: set default value for drop_remainder and max_rowsize
    dataloader = ds.batch(loader_config['batch_size'],
                    #per_batch_map=,
                    #input_columns=dataset_column_names,
                    drop_remainder=loader_config['drop_remainder'],
                    max_rowsize=loader_config['max_rowsize'])

    #steps_pre_epoch = dataset.get_dataset_size()
    return dataloader
