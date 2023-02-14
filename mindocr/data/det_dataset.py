from typing import Union, List
import random
import json
import os
import numpy as np
from addict import Dict
from mindocr.data.transforms.transforms_factory import DecodeImage
from mindocr.data.transforms.mz_db_transforms import MZSimpleNorm, ResizeByGrid, RandomScaleByShortSide, MZRandomCropData, MZMakeSegDetectionData, MZMakeBorderMap, MZRandomColorAdjust
from mindocr.data.transforms.iaa_augment import IaaAugment
from mindocr.data.transforms.random_crop_data import EastRandomCropData

class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        tmp = np.delete(pts, (np.argmin(s), np.argmax(s)), axis=0)
        diff = np.diff(np.array(tmp), axis=1)
        rect[1] = tmp[np.argmin(diff)]
        rect[3] = tmp[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes

    def __call__(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data

def create_transforms(transform_pipeline: List[dict], global_config=None):
    """
    create transforms based on the config

    Args:
        transform_pipeline (list of dict): a dict list defining the transformation pipeline. Fo each dict, the key is
            a transform class name, the value is a dict encoding the class init params and values.
            e.g. [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
    """
    assert isinstance(transform_pipeline, list), ('transform_pipeline config should be a list')
    transforms = []
    for transform_config in transform_pipeline:
        assert isinstance(transform_config,dict) and len(transform_config) == 1, "yaml format error in transforms"
        cls_name = list(transform_config.keys())[0]
        param = {} if transform_config[cls_name] is None else transform_config[cls_name]
        if global_config is not None:
            param.update(global_config)
        # TODO: assert undefined transform class
        transform = eval(cls_name)(**param)
        transforms.append(transform)
    return transforms


def run_transforms(data, transforms=None):
    """ transform """
    if transforms is None:
        transforms = []
    for transform in transforms:
        data = transform(data)
        if data is None:
            return None
    return data

def det_transforms_icdar15(is_train=True):
    transforms = [
        DecodeImage(img_mode='BGR', channel_first=False, to_float32=False),
        #ResizeByGrid(),
    ]
    return transforms

# TODO: inherit from BaseDataset
class DetDataset(object):
    """Data iterator for detection datasets including ICDAR15 dataset. 
    The annotaiton format is required to aligned to paddle, which can be done using the `converter.py` script.

    Args:
        dataset_config: dict with the following keys, 
            - data_dir, Required
            - label_file_list, Required
            - shuffle, Optional, if not given, shuffle = is_train
        transform_pipeline: list of dict, key - transform class name, value - a dict of param config.
                    e.g., [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
            -       if None, default transform pipeline for text detection will be taken.
        is_train
    Return:
        a dataset iterator, which generates a data item at each call.
        TODO: clarity data item format, dict / tuple.
            img_path, 
            image,
            polys,
            texts,
            ignore_tags, # 

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
    def __init__(self, data_dir: str, annot_files: Union[List, str] = '', sample_ratios: Union[List, float] = 1.0, 
            transform_pipeline: List[dict] = None, is_train: bool = True, shuffle: bool = None):
        self.data_dir = data_dir
        shuffle = shuffle if shuffle is not None else is_train
        
        # load data
        if annot_files == '':
            annot_files = os.path.join(data_dir, 'gt.txt')
        self.data_list = self.load_data_list(annot_files, sample_ratios, shuffle)

        # create transform
        if transform_pipeline is not None:
            self.transforms = create_transforms(transform_pipeline)
        else:
            self.transforms = det_transforms_icdar15()
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        
        # perform transformation on data
        # 1. accodring to config
        data = run_transforms(data, transforms=self.transforms)

        # 2. code embedded

        return data

    def load_data_list(self, annot_files: Union[str, List[str]], sample_ratios: Union[float, List] = 1.0, 
                shuffle: bool = False) -> List[dict]:
        '''Load annotations from an annotation file
        Args:
            annot_files: annotation file path(s)
            sample_ratios:
            shuffle: shuffle the data list

        Returns:
            List[dict]: A list of annotation dict, which contains keys: img_path, annot...
        '''
        if isinstance(annot_files, str):
            annot_files = [annot_files]
        if isinstance(sample_ratios, float):
            sample_ratios = [sample_ratios for _ in annot_files]

        # read annotation files
        data_lines = []
        for idx, annot_file in enumerate(annot_files):
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


def create_transforms_from_modelzoo_dbnet(is_train=True):
    if is_train:
        pipeline = [
                    {'DecodeImage': {'img_mode': 'BGR', 'to_float32': False}},
                    {'DetLabelEncode': None}, #TODO: check diff from model zoo get_boxes
                    {'ResizeByGrid': {'denominator': 32, 'transform_polys': False}},
                    {'RandomScaleByShortSide': {'short_side': 736}},
                    {'IaaAugment': {'augmenter_args':
                        [{
                            'type': 'Affine',
                            'args': {
                                'rotate': [-10, 10]
                            }
                        }, {
                            'type': 'Fliplr',
                            'args': {
                                'p': 0.5
                            }
                        },
                        ]
                        }
                    },
                    {'MZRandomCropData': 
                            {'max_tries':100, 
                            'min_crop_side_ratio': 0.1,
                            'crop_size': (640, 640)}},
                    {'ResizeByGrid': {'denominator': 32, 'transform_polys': True}},
                    {'MZMakeSegDetectionData', 
                            {'min_text_size': 8,
                            'shrink_ratio': 0.4}},
                    {'MZMakeBorderMap': {
                        'shrink_ratio': 0.4,
                        'thresh_min': 0.3,
                        'thresh_max': 0.7,
                    }},
                    {'MZRandomColorAdjust': {'brightness': 32.0 / 255, 'saturation':0.5, 'to_numpy':True}},
                    ]
    else:
        pass
    

if __name__=='__main__':
    data_dir = '/Users/Samit/Data/datasets/ic15/det/train'
    annot_file = '/Users/Samit/Data/datasets/ic15/det/train/train_icdar2015_label.txt'
    transform_dict = [
                    {'DecodeImage': {'img_mode': 'BGR', 'to_float32': False}},
                    {'DetLabelEncode': None}, #TODO: check diff from model zoo get_boxes
                    {'ResizeByGrid': {'denominator': 32}},
                    {'RandomScaleByShortSide': {'short_side': 736}},
                    #{'EastRandomCropData': {'max_tries': 100, 'min_crop_side_ratio': 0.1, 'size': (640, 640)}}
                    #{'MZRandomCropData': {'max_tries':100,  'min_crop_side_ratio': 0.1, 'crop_size': (640, 640)}},
                    {'ResizeByGrid': {'denominator': 32, 'transform_polys': True}},
                    # color_jitter w/ bgr or rgb param, convert to PIL?
                    {'MZRandomColorAdjust': {'brightness': 32.0 / 255, 'saturation':0.5, 'to_numpy':True}},
                    # reduce mean, div std
                    ]
    ds = DetDataset(data_dir, annot_file, 0.5, transform_pipeline=transform_dict, is_train=True, 
                    shuffle=True)

    #import matplotlib.pyplot as plt
    from mindocr.utils.visualize import show_img, draw_bboxes, show_imgs
    print('num data: ', len(ds))
    for i in range(1):
        data = ds.__getitem__(i)
        print(data['image'])
        print(data.keys())
        #print(data['image'])
        print(data['image'].shape)
        print(data['polys']) 
        print(data['texts']) 
        #show_img(data['image'], 'BGR')
        #result_img1 = draw_bboxes(data['ori_image'], data['polys'])
        result_img2 = draw_bboxes(data['image'], data['polys'])
        show_img(result_img2)
        #show_imgs([result_img1, result_img2])
