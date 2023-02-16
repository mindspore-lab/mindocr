import sys
sys.path.append('.')

import yaml
import glob
import pytest
import numpy as np
import mindspore as ms
import mindocr
from mindocr.data import build_dataset
from mindocr.data.det_dataset import DetDataset, transforms_dbnet_icdar15 
from mindspore import load_checkpoint, load_param_into_net

#@pytest.mark.parametrize('model_name', all_model_names)
def test_build_dataset():
    # TODO: download sample test data automatically
    data_dir = '/data/ocr_datasets/ic15/text_localization/train'
    annot_file = '/data/ocr_datasets/ic15/text_localization/train/train_icdar15_label.txt'

    dataset_config = {
            'type': 'DetDataset',
            'data_dir': data_dir,
            'label_files': [annot_file],
            'sample_ratios': [1.0],
            'shuffle': False,
            'transform_pipeline':
                [
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
                    {'ToCHWImage': None},
                    ],
            #'output_keys': ['image', 'threshold_map', 'threshold_mask', 'shrink_map', 'shrink_mask']
            'output_keys': ['img_path', 'image']
            }
    loader_config = {
            'shuffle': True, # TODO: tbc
            'batch_size': 8,
            'drop_remainder': True,
            'max_rowsize': 6,
            'num_workers': 2,
            }

    dl = build_dataset(dataset_config, loader_config, is_train=True)

    #batch = next(dl.create_tuple_iterator())
    batch = next(dl.create_dict_iterator())
    print(len(batch))


#@pytest.mark.parametrize('model_name', all_model_names)
def test_det_dataset():
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

if __name__ == '__main__':
    test_build_dataset()
    test_det_dataset()
