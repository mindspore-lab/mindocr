from __future__ import absolute_import
from __future__ import division

import sys
sys.path.append('.')

from typing import Union, List
import random
import os
import numpy as np
from addict import Dict

from mindocr.data.transforms.transforms_factory import create_transforms
from mindocr.data.transforms.transforms_factory import run_transforms

# TODO: inherit from BaseDataset
class DetDataset(object):
    """Data iterator for detection datasets including ICDAR15 dataset. 
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        data_dir, Required
        label_files, Required
        shuffle, Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        is_train
        output_keys (list): indicates the keys in data dict that are expected to output for dataloader. if None, all data keys will be used for return. 

    Returns:
        data (tuple): Depending on the transform pipeline, __get_item__ returns a subset of the following data. 
            - img_path (str), image path 
            - image (np.array), the format (CHW, RGB) is defined by the transform pipleine 
            - polys (np.array), shape (num_bboxes, num_points, 2)
            - texts (List),   
            - ignore_tags, # 
            - shrink_mask (np.array), binary mask for text region
            - shrink_map (np.array), 
            - threshold_mask (np.array), 
            - threshold_map (np.array), threshold map
        
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
            data_dir: str, 
            label_files: Union[List, str] = '', 
            sample_ratios: Union[List, float] = 1.0, 
            shuffle: bool = None,
            transform_pipeline: List[dict] = None, 
            output_keys: List[str] = None,
            is_train: bool = True, 
            **kwargs
            ):
        self.data_dir = data_dir
        shuffle = shuffle if shuffle is not None else is_train
        
        # load data
        if label_files == '':
            label_files = os.path.join(data_dir, 'gt.txt')
        self.data_list = self.load_data_list(label_files, sample_ratios, shuffle)

        # create transform
        if transform_pipeline is not None:
            self.transforms = create_transforms(transform_pipeline)
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

    def load_data_list(self, label_files: Union[str, List[str]], sample_ratios: Union[float, List] = 1.0, 
                shuffle: bool = False) -> List[dict]:
        '''Load annotations from an annotation file
        Args:
            label_files: annotation file path(s)
            sample_ratios:
            shuffle: shuffle the data list

        Returns:
            List[dict]: A list of annotation dict, which contains keys: img_path, annot...
        '''
        if isinstance(label_files, str):
            label_files = [label_files]
        if isinstance(sample_ratios, float):
            sample_ratios = [sample_ratios for _ in label_files]

        # read annotation files
        data_lines = []
        for idx, annot_file in enumerate(label_files):
            with open(annot_file, "r") as f:
                lines = f.readlines()
                if shuffle: #or sample_ratios[idx] < 1.0
                    # TODO: control random seed outside
                    #random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * sample_ratios[idx]))
                else:
                    lines = lines[:round(len(lines) * sample_ratios[idx])]
                data_lines.extend(lines)

        # parse each line of annotation
        data_list = []
        for data_line in data_lines:
            img_annot_dict = self._parse_annotation(data_line)
            data_list.append(img_annot_dict)

        return data_list

    def _parse_annotation(self, data_line: str) -> Union[dict, List[dict]]:
        '''
        Parse a data line str to a data item according to the predefined format 
        Returns:
            dict
                img_path:str, 
                polys: List of polygons/bboxes, each element is a numpy array of shape [N, 2], N is 4 for IC15 dataset 
                texts: list of strings for text instances)
        '''
        data = {}
        file_name, annot_str = data_line.strip().split('\t')
        img_path= os.path.join(self.data_dir, file_name)
        assert os.path.exists(img_path), "{} does not exist!".format(img_path)

        data['img_path'] = img_path 
        data['label'] = annot_str
        # the annotation string will be futher encoded in __get_item__ to match the network loss computation, using **LabelEncode defined in transform configs
        '''
        annot = json.load(annot_str)
        texts = []
        polys = []
        for instance in annot:
            texts.append(instance['transcription']) 
            polys.append(np.asarray(instance['points']))
        '''
        return data


def transforms_dbnet_icdar15(phase='train'):
    '''
    Get pre-defined transform config for dbnet on icdar15 dataset.
    Args:
        phase: train, eval, infer 

    Returns:
        list of dict for data transformation pipeline, which can be convert to functions by 'create_transforms' 
    '''
    if phase == 'train':
        pipeline = [
                    {'DecodeImage': {
                        'img_mode': 'BGR', 
                        'to_float32': False}},
                    {'DetLabelEncode': None}, 
                    {'MZResizeByGrid': {'denominator': 32, 'transform_polys': True}}, # prev in modelzoo, it doesn't transform polys
                    {'MZRandomScaleByShortSide': {'short_side': 736}},
                    {'IaaAugment': {'augmenter_args':
                        [{'type': 'Affine',
                            'args': {'rotate': [-10, 10]}
                        }, 
                        {'type': 'Fliplr',
                            'args': {
                                'p': 0.5}
                        },
                        ]}
                    },
                    {'MZRandomCropData': 
                            {'max_tries':100, 
                            'min_crop_side_ratio': 0.1,
                            'crop_size': (640, 640)}},
                    {'MZResizeByGrid': {'denominator': 32, 'transform_polys': True}},
                    #{'MakeShrinkMap': 
                    {'MZMakeSegDetectionData': 
                        {'min_text_size': 8, 'shrink_ratio': 0.4}},
                    #{'MakeBorderMap': 
                    {'MZMakeBorderMap': 
                        {'shrink_ratio': 0.4, 'thresh_min': 0.3, 'thresh_max': 0.7,
                    }},
                    {'MZRandomColorAdjust': {'brightness': 32.0 / 255, 'saturation':0.5, 'to_numpy':True}},
                    #{'MZIrregularNormToCHW': None},
                    {'NormalizeImage': {
                        'bgr_to_rgb': True,
                        'is_hwc': True,
                        'mean' : [123.675, 116.28, 103.53],
                        'std' : [58.395, 57.12, 57.375],
                        }
                    },
                    {'ToCHWImage': None}
                    ]

    elif phase=='eval':
        pipeline = [
                    {'DecodeImage': {'img_mode': 'BGR', 'to_float32': False}},
                    {'DetLabelEncode': None}, 
                    {'MZResizeByGrid': {'denominator': 32, 'transform_polys': True}}, # prev in modelzoo, it doesn't transform polys

                    {'MZScalePad': {'eval_size': [736, 1280]}},                     
                    {'NormalizeImage': {
                        'bgr_to_rgb': True,
                        'is_hwc': True,
                        'mean' : [123.675, 116.28, 103.53],
                        'std' : [58.395, 57.12, 57.375],
                        }
                    },
                    {'ToCHWImage': None}

                    ]
    else:
        pipeline = [
                    {'DecodeImage': {'img_mode': 'BGR', 'to_float32': False}},
                    {'MZResizeByGrid': {'denominator': 32, 'transform_polys': True}}, 
                    {'MZScalePad': {'eval_size': [736, 1280]}},                     
                    {'NormalizeImage': {
                        'bgr_to_rgb': True,
                        'is_hwc': True,
                        'mean' : [123.675, 116.28, 103.53],
                        'std' : [58.395, 57.12, 57.375],
                        }
                    },
                    {'ToCHWImage': None}
                    ]
    return pipeline



if __name__=='__main__':
    #data_dir = '/Users/Samit/Data/datasets/ic15/det/train'
    #annot_file = '/Users/Samit/Data/datasets/ic15/det/train/train_icdar2015_label.txt'
    data_dir = '/data/ocr_datasets/ic15/text_localization/train'
    annot_file = '/data/ocr_datasets/ic15/text_localization/train/train_icdar15_label.txt'
    transform_pipeline = transforms_dbnet_icdar15(phase='train') 
    ds = DetDataset(data_dir, annot_file, 0.5, transform_pipeline=transform_pipeline, is_train=True, shuffle=False)


    from mindocr.utils.visualize import show_img, draw_bboxes, show_imgs, recover_image
    print('num data: ', len(ds))
    for i in [223]:
        data_tuple = ds.__getitem__(i)
        
        # recover data from tuple to dict
        data = {k:data_tuple[i] for i, k in enumerate(ds.get_column_names())}

        print(data.keys())
        #print(data['image'])
        print(data['img_path'])
        print(data['image'].shape)
        print(data['polys']) 
        print(data['texts']) 
        #print(data['mask']) 
        #print(data['threshold_map']) 
        #print(data['threshold_mask']) 
        for k in data:
            print(k, data[k])
            if isinstance(data[k], np.ndarray):
                print(data[k].shape)
            
        #show_img(data['image'], 'BGR')
        #result_img1 = draw_bboxes(data['ori_image'], data['polys'])
        img_polys = draw_bboxes(recover_image(data['image']), data['polys'])
        #show_img(result_img2, show=False, save_path='/data/det_trans.png')
        
        mask_polys= draw_bboxes(data['shrink_mask'], data['polys'])
        thrmap_polys= draw_bboxes(data['threshold_map'], data['polys'])
        thrmask_polys= draw_bboxes(data['threshold_mask'], data['polys'])
        show_imgs([img_polys, mask_polys, thrmap_polys, thrmask_polys], show=False, save_path='/data/ocr_ic15_debug2.png')
        
        # TODO: check transformed image and label correctness
