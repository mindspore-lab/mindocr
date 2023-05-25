from mindocr import build_postprocess


class Postprocessor(object):
    def __init__(self, task='det', algo='DB', **kwargs):
        #algo = algo.lower()
        if task == 'det':
            if algo.startswith('DB'):
                postproc_cfg = dict(name='DBPostprocess',
                                    output_polygon=False,
                                    binary_thresh=0.3,
                                    box_thresh=0.6,
                                    max_candidates=1000,
                                    expand_ratio=1.5,
                                   )
            else:
                raise ValueError(f'No postprocess config defined for {algo}. Please check the algorithm name.')
        elif task=='rec':
            # TODO: update character_dict_path and use_space_char after CRNN trained using en_dict.txt released
            if algo.startswith('CRNN'):
                # TODO: allow users to input char dict path
                dict_path = 'mindocr/utils/dict/ch_dict.txt' if algo == 'CRNN_CH' else None
                postproc_cfg = dict(
                  name='RecCTCLabelDecode',
                  character_dict_path=dict_path,
                  use_space_char=False,
                )
            elif algo.startswith('RARE'):
                dict_path = 'mindocr/utils/dict/ch_dict.txt' if algo == 'RARE_CH' else None
                postproc_cfg = dict(
                  name='RecAttnLabelDecode',
                  character_dict_path=dict_path,
                  use_space_char=False,
                )

            else:
                raise ValueError(f'No postprocess config defined for {algo}. Please check the algorithm name.')

        postproc_cfg.update(kwargs)
        self.task = task
        self.postprocess = build_postprocess(postproc_cfg)

    def __call__(self, pred, data=None):
        '''
        Args:
            pred: network prediction
            data: (optional)
                preprocessed data, dict, which contains key `shape`
                    - shape: its values are [ori_img_h, ori_img_w, scale_h, scale_w]. scale_h, scale_w are needed to map the predicted polygons back to the orignal image shape.

        return:
            det_res: dict, elements:
                    - polys: shape [num_polys, num_points, 2], point coordinate definition: width (horizontal), height(vertical)
        '''

        output = self.postprocess(pred)

        if self.task == 'det':
            scale_h, scale_w = data['shape'][2:]
            # TODO: change return as dict in mindocr/postprocess
            # TODO: leave an option to do scale-back for polygons in build_postprocess
            polys, scores = output[0]
            if len(polys) > 0:
                if not isinstance(polys, list):
                    polys[:,:,0] = polys[:,:,0] / scale_w  #
                    polys[:,:,1] = polys[:,:,1] / scale_h
                else:
                    for i, poly in enumerate(polys):
                        polys[i][:,0] = polys[i][:,0] / scale_w
                        polys[i][:,1] = polys[i][:,1] / scale_h
            det_res = dict(polys=polys, scores=scores)

            return det_res
        elif self.task == 'rec':
            return output

