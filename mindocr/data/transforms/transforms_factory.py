'''
Create and run transformations from a config or predefined transformation pipeline
'''
from __future__ import absolute_import
from __future__ import division

from typing import List
import numpy as np

from .general_transforms import * 
from .det_transforms import *
from .rec_transforms import *
# TODO: merge transforms in modelzoo to det_transforms if verified to be correct
from .modelzoo_transforms import *
from .iaa_augment import *

__all__ = ['create_transforms', 'run_transforms', 'transforms_dbnet_icdar15']

# TODO: use class with __call__, to perform transformation
def create_transforms(transform_pipeline, global_config=None):
    """
    Create a squence of callable transforms.

    Args:
        transform_pipeline (List): list of callable instances or dicts where each key is a transformation class name, and its value are the args.
            e.g. [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                 [DecodeImage(img_mode='BGR')]

    Returns:
        list of data transformation functions
    """
    assert isinstance(transform_pipeline, list), (f'transform_pipeline config should be a list, but {type(transform_pipeline)} detected')

    transforms = []
    for transform_config in transform_pipeline:
        if isinstance(transform_config, dict):
            assert len(transform_config) == 1, "yaml format error in transforms"
            trans_name = list(transform_config.keys())[0]
            param = {} if transform_config[trans_name] is None else transform_config[trans_name]
            #  TODO: not each transform needs global config
            if global_config is not None:
                param.update(global_config)
            # TODO: assert undefined transform class

            #print(trans_name, param)
            transform = eval(trans_name)(**param)
            transforms.append(transform)
        elif callable(transform_config):
            transforms.append(transform_config)
        else:
            raise TypeError('transform_config must be a dict or a callable instance')
        #print(global_config)
    return transforms


def run_transforms(data, transforms=None, verbose=False):
    if transforms is None:
        transforms = []
    for i, transform in enumerate(transforms):
        if verbose:
            print(f'Trans {i}: ', transform)
            print(f'\t Input: ', {k: data[k].shape for k in data if isinstance(data[k], np.ndarray)})
        data = transform(data)
        if data is None:
            return None
    return data

# ---------------------- Predefined transform pipeline ------------------------------------
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
                    {'MZResizeByGrid': {'divisor': 32, 'transform_polys': True}}, # prev in modelzoo, it doesn't transform polys
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
                    {'MZResizeByGrid': {'divisor': 32, 'transform_polys': True}},
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
                    {'MZResizeByGrid': {'divisor': 32, 'transform_polys': True}}, # prev in modelzoo, it doesn't transform polys

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
                    {'MZResizeByGrid': {'divisor': 32, 'transform_polys': True}},
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