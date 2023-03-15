'''
Using a dummy images and ocr labels to test the train and eval pipeline.
Expected results: run pass, but loss may not decrease.

Example:
    pytest tests/st/test_train_eval_dummy.py

'''

import os
import subprocess
import sys
import glob

sys.path.append(".")

import pytest

from mindcv.utils.download import DownLoad


@pytest.mark.parametrize("task", ["det", "rec"])
def test_train_eval(task):
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
    
    # ---------------- test running train.py using the toy data ---------
    if task == 'det':
        config_fp = 'tests/st/det_db_test.yaml'
    elif task=='rec':
        config_fp = 'tests/st/rec_crnn_test.yaml'

    #if os.path.exists(ckpt_dir):
    #    os.system(f"rm {ckpt_dir} -rf")


    cmd = (
        f"python tools/train.py --config {config_fp}"
    )

    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Training fails"

    # --------- Test running validate.py using the trained model ------------- #
    # begin_ckpt = os.path.join(ckpt_dir, f'{model}-1_1.ckpt')
    cmd = (
        f"python tools/eval.py --config {config_fp}"
    )
    # ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    print(f"Running command: \n{cmd}")
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = p.communicate()
    # assert ret==0, 'Validation fails'
    print(out)

    p.kill()
    '''
    if check_acc:
        res = out.decode()
        idx = res.find("Accuracy")
        acc = res[idx:].split(",")[0].split(":")[1]
        print("Val acc: ", acc)
        assert float(acc) > 0.5, "Acc is too low"
    '''

if __name__ == '__main__':
    #test_train('det')
    test_train_eval('rec')
