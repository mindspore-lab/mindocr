from typing import List
import os
import multiprocessing
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
    Build dataset

    Args:
        dataset_config (dict): dataset reading and processing configuartion containing keys:
            - type: dataset type, 'DetDataset', 'RecDataset'
            - dataset_root (str): the root directory to store the (multiple) dataset(s)
            - data_dir (Union[str, List[str]]): directory to the data, which is a subfolder path related to `dataset_root`. For multiple datasets, it is a list of subfolder paths.
            - label_file (Union[str, List[str]]): file path to the annotation related to the `dataset_root`. For multiple datasets, it is a list of relative file paths.
            - transform_pipeline (list[dict]): each element corresponds to a transform operation on image and/or label

        loader_config (dict): dataloader configuration containing keys:
            - batch_size: batch size for data loader
            - drop_remainder: whether to drop the data in the last batch when the total of data can not be divided by the batch_size
        num_shards: num of devices for distributed mode
        shard_id: device id
        is_train: whether it is in training stage

    Return:
        data_loader (Dataset): dataloader to generate data batch

    Notes:
        - The main data process pipeline in MindSpore contains 3 parts: 1) load data files and generate source dataset, 2) perform per-data-row mapping such as image augmentation, 3) generate batch and apply batch mapping. 
        - Each of the three steps supports multiprocess. Detailed machenism can be seen in https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore.dataset.html
        - A data row is a data tuple item containing multiple elements such as (image_i, mask_i, label_i). A data column corresponds to an element in the tuple like 'image', 'label'. 
        - The total number of `num_parallel_workers` used for data loading and processing should not be larger than the maximum threads of the CPU. Otherwise, it will lead to resource competing overhead. Especially for distributed training, `num_parallel_workers` should not be too large to avoid thread competition. 
    '''
    ## check and process dataset_root, data_dir, and label_file.
    if 'dataset_root' in dataset_config:
        if isinstance(dataset_config['data_dir'], str):
            dataset_config['data_dir'] = os.path.join(dataset_config['dataset_root'], dataset_config['data_dir']) # to absolute path
        else:
            dataset_config['data_dir'] = [os.path.join(dataset_config['dataset_root'], dd) for dd in dataset_config['data_dir']]

        if 'label_file' in dataset_config:
            if isinstance(dataset_config['label_file'], str):
                dataset_config['label_file'] = os.path.join(dataset_config['dataset_root'], dataset_config['label_file'])
            else:
                dataset_config['label_file'] = [os.path.join(dataset_config['dataset_root'], lf) for lf in dataset_config['label_file']]

    # build datasets
    dataset_class_name = dataset_config.pop('type')
    assert dataset_class_name in supported_dataset_types, "Invalid dataset name"
    dataset_class = eval(dataset_class_name)

    dataset_args = dict(is_train=is_train, **dataset_config)
    dataset = dataset_class(**dataset_args)

    # create batch loader
    dataset_column_names = dataset.get_column_names()
    print('==> Dataset columns: \n\t', dataset_column_names)

    # TODO: find optimal setting automatically according to num of CPU cores
    num_workers = loader_config.get("num_workers", 8) # Number of subprocesses used to fetch the dataset/map data row/gen batch in parallel
    cores = multiprocessing.cpu_count()
    num_devices = 1 if num_shards is None else num_shards 
    if num_workers > int(cores / num_devices):
        num_workers = int(cores / num_devices)
        print('WARNING: num_workers is adjusted to {num_workers}, to fit {cores} CPU cores shared for {num_devices} devices')

    prefetch_size = loader_config.get("prefetch_size", 16) # the length of the cache queue in the data pipeline for each worker, used to reduce waiting time. Larger value leads to more memory consumption. Default: 16 
    max_rowsize =  loader_config.get("max_rowsize", 64) # MB of shared memory between processes to copy data
    
    ms.dataset.config.set_prefetch_size(prefetch_size)  
    #print('Prefetch size: ', ms.dataset.config.get_prefetch_size())

    # auto tune num_workers, prefetch. (This conflicts the profiler)
    #ms.dataset.config.set_autotune_interval(5)
    #ms.dataset.config.set_enable_autotune(True, "./dataproc_autotune_out")  

    # 1. generate source dataset (source w.r.t. the dataset.map pipeline) based on python callable numpy dataset in parallel 
    ds = ms.dataset.GeneratorDataset(
                    dataset,
                    column_names=dataset_column_names,
                    num_parallel_workers=num_workers,
                    num_shards=num_shards,
                    shard_id=shard_id,
                    python_multiprocessing=True, # keep True to improve performace for heavy computation.
                    max_rowsize =max_rowsize,
                    shuffle=loader_config['shuffle'],
                    )

    # 2. per-data-item mapping (high-performance transformation)
    #ds = ds.map(operations=transform_list, input_columns=['image', 'label'], num_parallel_workers=8, python_multiprocessing=True)

    
    # 3. get batch of dataset by collecting batch_size consecutive data rows and apply batch operations 
    drop_remainder = loader_config.get('drop_remainder', is_train)
    if is_train and drop_remainder == False:
        print('WARNING: drop_remainder should be True for training, otherwise the last batch may lead to training fail.')
    dataloader = ds.batch(
                    loader_config['batch_size'],
                    drop_remainder=drop_remainder,
                    num_parallel_workers=min(num_workers//2, 1), # set small value since it is lite computation. 
                    #input_columns=input_columns,
                    #output_columns=batch_column,
                    #per_batch_map=per_batch_map, # uncommet to use inner-batch transformation
                    )

    #steps_pre_epoch = dataset.get_dataset_size()
    return dataloader
