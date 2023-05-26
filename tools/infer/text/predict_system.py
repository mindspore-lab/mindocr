'''
Text detection and recognition inference

Example:
    $ python tools/infer/text/predict_system.py --image_dir {path_to_img_file} --det_algorithm DB++ --rec_algorithm CRNN
    $ python tools/infer/text/predict_system.py --image_dir {path_to_img_dir} --det_algorithm DB++ --rec_algorithm CRNN_CH
'''

import os
import sys
import argparse
from typing import Union
import numpy as np
import cv2
from time import time
import mindspore as ms
import json

from predict_det import TextDetector
from predict_rec import TextRecognizer
from utils import crop_text_region, get_image_paths, get_ckpt_file
from config import parse_args

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from mindocr.utils.visualize import visualize


class TextSystem(object):
    def __init__(self, args):
        self.text_detect = TextDetector(args)
        self.text_recognize = TextRecognizer(args)

        self.box_type = args.det_box_type
        self.drop_score = args.drop_score
        self.save_crop_res = args.save_crop_res
        self.crop_res_save_dir = args.crop_res_save_dir
        if self.save_crop_res:
            os.makedirs(self.crop_res_save_dir, exist_ok=True)
        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        self.vis_font_path = args.vis_font_path

    def __call__(self, img_or_path: Union[str, np.ndarray], do_visualize=True):
        '''
        Detect and recognize texts in an image

        Args:
            img_or_path (str or np.ndarray): path to image or image rgb values as a numpy array

        Return:
            boxes (list): detected text boxes, in shape [num_boxes, num_points, 2], where the point coordinate (x, y) follows: x - horizontal (image width direction), y - vertical (image height)
            texts (list[tuple]): list of (text, score) where text is the recognized text string for each box, and score is the confidence score.
            time_profile (dict): record the time cost for each sub-task.
        '''
        assert isinstance(img_or_path, str) or isinstance(img_or_path, np.ndarray), "Input must be string of path to the image or numpy array of the image rgb values."
        fn = os.path.basename(img_or_path).split('.')[0] if isinstance(img_or_path, str) else 'img'

        time_profile = {}
        start = time()

        # detect text regions on an image
        det_res, data = self.text_detect(img_or_path, do_visualize=False)
        time_profile['det'] = time() - start
        #print(det_res)
        polys = det_res['polys'].copy()
        print(f"INFO: Num detected text boxes: {len(polys)}\nDet time: ", time_profile['det'])

        # crop text regions
        crops = []
        for i in range(len(polys)):
            poly = polys[i].astype(np.float32)
            cropped_img = crop_text_region(data['image_ori'], poly, box_type=self.box_type)
            crops.append(cropped_img)
            #print('Crop ', i, cropped_img.shape)

            if self.save_crop_res:
                cv2.imwrite(os.path.join(output_dir, f"{fn}_crop_{i}.jpg"), cropped_img)
        #show_imgs(crops, is_bgr_img=False)

        # recognize cropped images
        rs = time()
        rec_res_all_crops = self.text_recognize(crops, do_visualize=False)
        time_profile['rec'] = time() - rs

        print('INFO: Recognized texts: \n' +
              "\n".join([f"{text}\t{score}" for text, score in rec_res_all_crops]) +
              '\nRec time: ', time_profile['rec'])

        # filter out low-score texts and merge detection and recognition results
        boxes, text_scores = [], []
        for i in range(len(polys)):
            box = det_res['polys'][i]
            box_score = det_res['scores'][i]
            text = rec_res_all_crops[i][0]
            text_score = rec_res_all_crops[i][1]
            if text_score >= self.drop_score:
                boxes.append(box)
                text_scores.append((text, text_score))

        time_profile['all'] = time() - start

        # visualize the overall result
        if do_visualize:
            vst = time()
            vis_fp = os.path.join(self.vis_dir, fn+'_res.png')
            # TODO: improve vis for leaning texts
            visualize(data['image_ori'],
                      boxes,
                      texts=[x[0] for x in text_scores],
                      vis_font_path=self.vis_font_path,
                      display=False,
                      save_path=vis_fp,
                      draw_texts_on_blank_page=False) # NOTE: set as you want
            time_profile['vis'] = time() - vst
        return boxes, text_scores, time_profile

def save_res(boxes_all, text_scores_all, img_paths, save_path="system_results.txt"):
    lines = []
    for i, img_path in enumerate(img_paths):
        #fn = os.path.basename(img_path).split('.')[0]
        boxes = boxes_all[i]
        text_scores = text_scores_all[i]

        res = [] # result for current image
        for j in range(len(boxes)):
            res.append({"transcription": text_scores[j][0],
                    "points": np.array(boxes[j]).astype(np.int32).tolist(),
                })

        img_res_str = os.path.basename(img_path) +  "\t" + json.dumps(res, ensure_ascii=False) + "\n"
        lines.append(img_res_str)

    with open(save_path, 'w') as f:
        f.writelines(lines)
        f.close()


def main():
    # parse args
    args = parse_args()
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)

    # uncomment it to quick test the infer FPS
    #img_paths = img_paths[:10]

    ms.set_context(mode=args.mode)

    # init text system with detector and recognizer
    text_spot = TextSystem(args)

    # warmup
    if args.warmup:
        for i in range(2):
            text_spot(img_paths[0], do_visualize=False)

    # run
    tot_time = {} #{'det': 0, 'rec': 0, 'all': 0}
    boxes_all, text_scores_all = [], []
    for i, img_path in enumerate(img_paths):
        print(f'\nINFO: Infering [{i+1}/{len(img_paths)}]: ', img_path)
        boxes, text_scores, time_prof = text_spot(img_path, do_visualize=args.visualize_output)
        boxes_all.append(boxes)
        text_scores_all.append(text_scores)

        for k in time_prof:
            if k not in tot_time:
                tot_time[k] = time_prof[k]
            else:
                tot_time[k] += time_prof[k]

    fps = len(img_paths) / tot_time['all']
    print('Total time:', tot_time['all'])
    print('Average FPS: ', fps)
    avg_time = {k: tot_time[k]/len(img_paths) for k in tot_time}
    print('Averge time cost: ', avg_time)

    # save result
    save_res(boxes_all, text_scores_all, img_paths,
             save_path=os.path.join(save_dir, 'system_results.txt'))
    print('Done! Results saved in ', os.path.join(save_dir, 'system_results.txt'))

if __name__ == '__main__':
    main()

