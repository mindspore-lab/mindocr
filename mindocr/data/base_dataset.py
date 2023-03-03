#from __future__ import absolute_import
from __future__ import division

from typing import Union, List
import random
import os

from .transforms.transforms_factory import create_transforms, run_transforms

class BaseDataset(object):
    """Data iterator for ocr datasets including ICDAR15 dataset. 
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        is_train: 
        data_dir: 
        label_files, 
        shuffle, Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        output_keys (list): required, indicates the keys in data dict that are expected to output for dataloader. if None, all data keys will be used for return. 
        global_config: additional info, used in data transformation, possible keys:
            - character_dict_path
            

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a tuple for the specified data item. 
        You can specify the `output_keys` arg to order the output data for dataloader.

    Notes: 
        1. Dataset file structure should follow:
            data_dir
            ├── images/
            │  │   ├── 000001.jpg
            │  │   ├── 000002.jpg
            │  │   ├── ... 
            ├── annotation_file.txt
        2. Annotation format should follow (img path and annotation are seperated by tab):
            # image path relative to the data_dir\timage annotation information encoded by json.dumps
            ch4_test_images/img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]   
    """
    def __init__(self, 
            is_train: bool = True, 
            data_dir: str = '', 
            label_files: Union[List, str] = '', 
            sample_ratios: Union[List, float] = 1.0, 
            shuffle: bool = None,
            transform_pipeline: List[dict] = None, 
            output_keys: List[str] = None,
            #global_config: dict = None,
            **kwargs
            ):
        self.data_dir = data_dir
        assert isinstance(shuffle, bool), f'type error of {shuffle}'
        shuffle = shuffle if shuffle is not None else is_train
        
        # load data
        #if label_files == '':
        #    label_files = os.path.join(data_dir, 'gt.txt')
        self.data_list = self.load_data_list(label_files, sample_ratios, shuffle)


        # create transform
        if transform_pipeline is not None:
            self.transforms = create_transforms(transform_pipeline) #, global_config=global_config)
        else:
            raise ValueError('No transform pipeline is specified!')

        # prefetch the data keys, to fit GeneratorDataset
        _data = self.data_list[0]
        _data = run_transforms(_data, transforms=self.transforms)
        _available_keys = list(_data.keys())
        
        if output_keys is None:
            self.output_keys = _available_keys   
        else:
            self.output_keys = []
            for k in output_keys:
                if k in _data:
                    self.output_keys.append(k)
                else:
                    raise ValueError(f'Key {k} does not exist in data (available keys: {_data.keys()}). Please check the name or the completeness transformation pipeline.')
                    
    def __len__(self):
        return len(self.data_list)

    def get_column_names(self):
        ''' return list of names for the output data tuples'''
        return self.output_keys

    def __getitem__(self, index):
        data = self.data_list[index]
        
        # perform transformation on data
        data = run_transforms(data, transforms=self.transforms)
            
        output_tuple = tuple(data[k] for k in self.output_keys) 

        return output_tuple

    def load_data_list(self, label_files: Union[str, List[str]], sample_ratios: Union[float, List] = 1.0,  shuffle: bool = False, **kwargs) -> List[dict]:
        ''' Load data list from label_files which contains infomation of image paths and annotations 
        Args:
            label_files: annotation file path(s)
            sample_ratios: sample ratio for data items in each annotation file
            shuffle: shuffle the data list

        Returns:
            data (List[dict]): A list of annotation dict, which contains keys: img_path, annot...
        '''
        if isinstance(label_files, str):
            label_files = [label_files]
        if isinstance(sample_ratios, float):
            sample_ratios = [sample_ratios for _ in label_files]

        # read annotation files
        data_lines = []
        for idx, annot_file in enumerate(label_files):
            with open(annot_file, "r", encoding='utf-8') as f:
                lines = f.readlines()
                if shuffle:
                    lines = random.sample(lines,
                                          round(len(lines) * sample_ratios[idx]))
                else:
                    lines = lines[:round(len(lines) * sample_ratios[idx])]
                data_lines.extend(lines)
                #print(lines[:5])
        # print(data_lines)
        # parse each line of annotation
        data_list = []
        for data_line in data_lines:
            img_annot_dict = self._parse_annotation(data_line)
            data_list.append(img_annot_dict)

        return data_list
        
    def _parse_annotation(self, data_line: str) -> Union[dict, List[dict]]:
        '''
        Initially parse a data line string into a data dict containing input (img path) and label info (json dump). The label info will be encoded in transformation. 
        '''
        data = {}
        file_name, annot_str = data_line.strip().split('\t')
        img_path= os.path.join(self.data_dir, file_name)

        #if os.path.exists(img_path):
        #    print('------->', img_path)
        assert os.path.exists(img_path), "{} does not exist!".format(img_path)

        data['img_path'] = img_path 
        data['label'] = annot_str
        # the annotation string will be futher encoded in __get_item__ to match the network loss computation, using **LabelEncode defined in transform configs
        
        return data
