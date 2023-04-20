from _common import gen_dummpy_data, update_config_for_CI

import sys
sys.path.append('.')

import yaml
import pytest
import time

import mindspore as ms
import mindocr
from mindocr.data import build_dataset
from mindocr.utils.visualize import show_img, draw_bboxes, recover_image


@pytest.mark.parametrize('task', ['det', 'rec'])
@pytest.mark.parametrize('phase', ['train', 'eval'])
def test_build_dataset(task, phase, verbose=False, visualize=False):
    # modify ocr predefined yaml for minimum test
    if task == 'det':
        config_fp = 'configs/det/dbnet/db_r50_icdar15.yaml'
    elif task=='rec':
        config_fp = 'configs/rec/crnn/crnn_icdar15.yaml'

    data_dir = gen_dummpy_data(task)
    yaml_fp = update_config_for_CI(config_fp, task)

    with open(yaml_fp) as fp:
        cfg = yaml.safe_load(fp)

    dataset_config = cfg[phase]['dataset']
    loader_config = cfg[phase]['loader']

    dl = build_dataset(dataset_config, loader_config, is_train=(phase=='train'))
    #dl.repeat(300)
    num_batches = dl.get_dataset_size()

    ms.set_context(mode=0)
    #batch = next(dl.create_tuple_iterator())
    num_tries = 100
    start = time.time()
    times = []
    iterator = dl.create_dict_iterator()
    for i, batch in enumerate(iterator):
        times.append(time.time()-start)
        if i >= num_tries:
            break

        if verbose:
            for k,v in batch.items():
                print(k, v.shape)
                if len(v.shape)<=2:
                    print(v)

        if (i == num_tries -1) and visualize:
            if task == 'det' and phase == 'eval':
                img = batch['image'][0].asnumpy()
                polys = batch['polys'][0].asnumpy()
                img_polys = draw_bboxes(recover_image(img), polys)
                show_img(img_polys)

        start = time.time()

    WU = 2
    tot = sum(times[WU:]) # skip warmup
    mean = tot / (num_tries-WU)
    print('Avg batch loading time: ', mean)


if __name__ == '__main__':
    #test_build_dataset(task='rec', phase='train', visualize=False)
    test_build_dataset(task='det', phase='train', visualize=False)
