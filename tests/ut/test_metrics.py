import sys
sys.path.append('.')

import numpy as np

from mindocr.data.det_dataset import DetDataset
from mindocr.postprocess.det_postprocess import DBPostprocess
from mindocr.metrics.det_metrics import DetMetric


def test_det_metric():
    # TODO: gen by DetDataset
    data = np.load('./det_db_label_samples.npz')
    polys, bmap, _, _, texts, ignore_tags = data['polys'], data['shrink_map'], data['threshold_map'], data['threshold_mask'], data['texts'], data['ignore_tags']
    polys = np.array([polys])
    bmap = np.array([bmap])
    ignore_tags  = np.array([ignore_tags])
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
    preds = proc({'binary': bmap})
    
    m = DetMetric() 
    m.update(preds, (polys, ignore_tags))

    res = m.eval()
    print(res)
    

if __name__=='__main__':
    test_det_metric()
