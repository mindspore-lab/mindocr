from typing import Union, List
import random
import json
import os
import numpy as np
from addict import Dict


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

# TODO: inherit from BaseDataset
class DetDataset():
    """Dataset iterator for detection task, including ICDAR15 dataset. 
    Args:
    dataset_config: required keys
        - data_dir:
        - label_file_list:
    transform_config: 
        - 
    
    Return:
        dataset iterator
        
    Note: 
        Folder and file structure should be like:
            data_dir
            ├── images/
            │  │   ├── 000001.jpg
            │  │   ├── 000002.jpg
            │  │   ├── ... 
            ├── annotation_file.txt


        Annotation format should follow:
            # Relative image path\tImage annotation information encoded by json.dumps
            ch4_test_images/img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]   

        It does not support LMDB dataset, Please use `LMDBDataset` instead.
    """
    def __init__(self, dataset_config: dict, transform_config :dict=None, is_train=True):
        load_data()

        self.data_dir = dataset_config['data_dir']
        
        self.data_list = load_data_list(dataset_config['label_file_list'], 
                dataset_config['ratio_list'], 
                dataset_config['shuffle'])
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        
        # transforms


        # adapt to mindspore data generator
        
        return data
    
    def load_data_list(self, 
            annot_files: Union[str, List(str)], 
            sample_ratios: List=[], 
            shuffle: bool=False) -> List[dict]:
	"""Load annotations from an annotation file
        Args:
            annot_files: annotation file path(s)
            shuffle: shuffle the data list

        Returns:
            List[dict]: A list of annotation dict, which contains keys: img_path, annot... 
        """
        if isinstance(annot_files, str):
            annot_fils = [annot_files]
        if sample_ratios = []:
            sample_ratios = [1.0 for _ in annot_files]
        
        # read annotation files 
        data_lines = []
        for idx, annot_file in enumerate(annot_files):
            with open(annot_file, "r") as f:
                lines = f.readlines()
                if shuffle or sample_ratios[idx] < 1.0:
                    # TODO: control random seed outside
                    #random.seed(self.seed)
                    lines = random.sample(lines,
                                          round(len(lines) * sample_ratios[idx]))
                data_lines.extend(lines)

        # parse each line of annotation
        data_list = []
        for data_line in datalines:
            img_annot_dict = self._parse_annotation(data_line)
            data_list.append(img_annot_dict)

        return data_list

    def _parse_annotation(self, data_line: str) -> Union[dict, List[dict]):
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
        data['img_path'] = os.path.join(self.data_dir, file_name)
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
         
            
        
        
        


        
    def load_data(self, data_path: str) -> list:
        """
       
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a dict, containing 'img_path','img_name','text_polys','texts','ignore_tags'
        """
        raise NotImplementedError




    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # Getting
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = get_bboxes(gt_path, self.config)

        # Random Augment
        if self.isTrain and self.config.train.is_transform:
            img, polys = self.ra.random_scale(img, polys, self.config.dataset.short_side)
            img, polys = self.ra.random_rotate(img, polys, self.config.dataset.random_angle)
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))
        img, polys = resize(img, polys, isTrain=self.isTrain)

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)
            img, polys = scale_pad(img, polys, self.config.eval.eval_size)

        # Show Images
        if self.config.dataset.is_show:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0]*255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask*255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map*255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask*255)

        # Random Colorize
        if self.isTrain and self.config.train.is_transform:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        return original, img, polys, dontcare


