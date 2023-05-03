import sys
sys.path.append('.')
import os
import glob
import yaml
from mindocr.models.backbones.mindcv_models.download import DownLoad

def gen_dummpy_data(task):
    # prepare dummy images
    data_dir = "data/Canidae"
    dataset_url = (
        "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/intermediate/Canidae_data.zip"
    )
    if not os.path.exists(data_dir):
        DownLoad().download_and_extract_archive(dataset_url, "./")

    # prepare dummy labels
    for split in ['train', 'val']:
        label_path = f'tests/st/dummy_labels/{task}_{split}_gt.txt'
        image_dir = f'{data_dir}/{split}/dogs'
        new_label_path = f'data/Canidae/{split}/{task}_gt.txt'
        img_paths = glob.glob(os.path.join(image_dir, '*.JPEG'))
        #print(len(img_paths))
        with open(new_label_path, 'w') as f_w:
            with open(label_path, 'r') as f_r:
                i = 0
                for line in f_r:
                    _, label = line.strip().split('\t')
                    #print(i)
                    img_name = os.path.basename(img_paths[i])
                    new_img_label = img_name + '\t' + label
                    f_w.write(new_img_label + '\n')
                    i += 1
        print(f'Dummpy annotation file is generated in {new_label_path}')

    return data_dir

def update_config_for_CI(config_fp, task, val_while_train=False, gradient_accumulation_steps=1, clip_grad=False, grouping_strategy=None, ema=False):
    with open(config_fp) as fp:
        config = yaml.safe_load(fp)
        config['system']['distribute'] = False
        config['system']['val_while_train'] = val_while_train
        config['optimizer']['grouping_strategy'] = grouping_strategy
        #if 'common' in config:
        #    config['batch_size'] = 8
        config['train']['dataset_sink_mode'] = False
        config['train']['gradient_accumulation_steps'] = gradient_accumulation_steps
        config['train']['clip_grad'] = clip_grad
        config['train']['ema'] =ema

        config['train']['dataset']['dataset_root'] = 'data/Canidae/'
        config['train']['dataset']['data_dir'] = 'train/dogs'
        config['train']['dataset']['label_file'] = f'train/{task}_gt.txt'
        config['train']['dataset']['sample_ratio'] = 0.1 # TODO: 120 training samples in total, don't be larger than batchsize after sampling
        config['train']['loader']['num_workers'] = 1 # github server only support 2 workers at most
        #if config['train']['loader']['batch_size'] > 120:
        config['train']['loader']['batch_size'] = 2 # to save memory
        config['train']['loader']['max_rowsize'] = 16 # to save memory
        config['train']['loader']['prefetch_size'] = 2 # to save memory
        if 'common' in config:
            config['common']['batch_size'] = 2
        if 'batch_size' in config['loss']:
            config['loss']['batch_size'] = 2

        config['eval']['dataset']['dataset_root'] = 'data/Canidae/'
        config['eval']['dataset']['data_dir'] = 'val/dogs'
        config['eval']['dataset']['label_file'] = f'val/{task}_gt.txt'
        config['eval']['dataset']['sample_ratio'] = 0.1
        config['eval']['loader']['num_workers'] = 1 # github server only support 2 workers at most
        config['eval']['loader']['batch_size'] = 1
        config['eval']['loader']['max_rowsize'] = 16 # to save memory
        config['eval']['loader']['prefetch_size'] = 2 # to save memory

        config['eval']['ckpt_load_path'] = os.path.join(config['train']['ckpt_save_dir'], 'best.ckpt')

        config['scheduler']['num_epochs'] = 2
        config['scheduler']['warmup_epochs'] = 1
        config['scheduler']['decay_epochs'] = 1

        dummpy_config_fp =os.path.join('tests/st', os.path.basename(config_fp.replace('.yaml', '_dummpy.yaml')))
        with open(dummpy_config_fp, 'w') as f:
            args_text = yaml.safe_dump(config, default_flow_style=False, sort_keys=False)
            f.write(args_text)
            print('Genearted yaml: ')
            print(args_text)

    return dummpy_config_fp


