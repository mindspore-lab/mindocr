"""
"""
from typing import Union

import numpy as np

import mindspore as ms
from mindspore import Tensor

from ..models.utils.abinet_layers import CharsetMapper

__all__ = ["ABINetLabelDecode"]


class ABINetLabelDecode(object):
    def __init__(
        self,
        lower=False,
    ):
        self.space_idx = None
        self.lower = lower
        self.charset = CharsetMapper(max_length=26)

    def decode(self, logit):
        """Greed decode"""
        # TODO: test running time and decode on GPU
        ms_softmax = ms.ops.Softmax(axis=2)
        out = ms_softmax(logit)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(axis=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(axis=1)[0])
            pt_lengths.append(min(len(text) + 1, 26))  # one for end-token
        ms_stack = ms.ops.Stack()
        pt_scores = ms_stack(pt_scores)
        pt_lengths = ms.Tensor(pt_lengths, dtype=ms.int64)

        return pt_text, pt_scores, pt_lengths

    def __call__(self, preds: Union[Tensor, np.ndarray], labels=None, **kwargs):
        """
        Args:
            preds (Union[Tensor, np.ndarray]): network prediction, class probabilities in shape [BS, W, num_classes],
                where W is the sequence length.
            labels: optional
        Return:
            texts (List[Tuple]): list of string

        """
        logits = preds
        pt_text, pt_scores, pt_lengths_ = self.decode(logits)
        pt_text = [self.charset.trim(t) for t in pt_text]
        return {"texts": pt_text}

    def _get_output(self, last_output):
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                for i in range(len(res)):
                    if len(res) == 3:
                        if res[i]["name"] == "alignment":
                            output = res[i]
        else:
            output = last_output
        return output

    def _update_output(self, last_output, items):
        if isinstance(last_output, (tuple, list)):
            res = last_output
            if res[3] == "alignment":
                res.update(items)
        else:
            last_output.update(items)
        return last_output
