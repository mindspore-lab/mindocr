import sys
sys.path.append('.')

import numpy as np

from mindocr.data.det_dataset import DetDataset
from mindocr.postprocess.det_postprocess import DBPostprocess


def test_det_db_postprocess():
    # TODO: gen by DetDataset
    data = np.load('./det_db_label_samples.npz')
    polys, bmap, _, _, texts, ignore_tags = data['polys'], data['shrink_map'], data['threshold_map'], data['threshold_mask'], data['texts'], data['ignore_tags']
    bmap = np.array([bmap])
    print('GT polys: ', polys)
    print('ignore_tags', ignore_tags)
    print('texts', texts)

    proc = DBPostprocess(thresh=0.3,  
                box_thresh=0.55, 
                max_candidates=1000, 
                unclip_ratio=1.5,
                region_type='quad', 
                dest='binary',
                score_mode='fast')
    res = proc({'binary': bmap})
    print(res)

if __name__=='__main__':
    test_det_db_postprocess()
