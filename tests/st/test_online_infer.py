import os
import subprocess
import sys
import glob
import yaml
import pytest
import cv2
import numpy as np

sys.path.append(".")
from mindocr.utils.visualize import draw_texts_with_boxes

TEXTS_2 = ['mindspore ai', '乘风破浪会有时，直挂云帆济沧海。']
BOXES_2 = [ [[30, 30], [30, 130], [70, 130], [70, 30]],
            [[100, 200], [100, 350], [250, 250], [250, 200]],
           ]

def _gen_text_image(texts=TEXTS_2,
                    boxes=BOXES_2,
                    save_fp='gen_img.jpg'
                    ):
    boxes = np.array(boxes, dtype=int)
    shape = [int(boxes[:,:,0].max() *1.15), int((boxes[:,:,1].max()*1.15))]

    bg = np.ones([shape[0], shape[1], 3], dtype=np.uint8) * 255
    hide_boxes = True
    text_inside_box = True
    font_size = 16
    text_color = (0, 0, 0)

    text_vis = draw_texts_with_boxes(
                            bg,
                            boxes,
                            texts,
                            text_color=text_color,
                            font_path='docs/fonts/simfang.ttf',
                            font_size=font_size,
                            hide_boxes=hide_boxes,
                            text_inside_box=text_inside_box)

    cv2.imwrite(save_fp, text_vis)

    return save_fp

det_img_fp = _gen_text_image(save_fp='gen_det_input.jpg')
rec_img_fp = _gen_text_image([TEXTS_2[0]], [BOXES_2[0]], 'gen_rec_input.jpg')

def test_det_infer():
    algo = 'DB'
    cmd = (
        f"python tools/infer/text/predict_det.py --image_dir {det_img_fp} --det_algorithm {algo} --draw_img_save_dir ./infer_test --visualize_output True"
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Det inference fails"

def test_rec_infer():
    algo = 'CRNN'
    cmd = (
        f"python tools/infer/text/predict_rec.py --image_dir {rec_img_fp} --rec_algorithm {algo} --draw_img_save_dir  ./infer_test"
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "Rec inference fails"


def test_system_infer():
    det_algo = 'DB'
    rec_algo = 'CRNN_CH'
    cmd = (
        f"python tools/infer/text/predict_system.py --image_dir {det_img_fp} --det_algorithm {det_algo} --rec_algorithm {rec_algo} --draw_img_save_dir ./infer_test --visualize_output True"
    )
    print(f"Running command: \n{cmd}")
    ret = subprocess.call(cmd.split(), stdout=sys.stdout, stderr=sys.stderr)
    assert ret == 0, "System inference fails"
