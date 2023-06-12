from mindspore import Tensor

__all__ = ["ClsPostProcess"]


class ClsPostProcess(object):
    """Map the predicted index back to orignal format (angle)."""

    def __init__(self, label_list=None, **kwargs):
        assert (
            label_list is not None
        ), "`label_list` should not be None. Please set it in 'postprocess' section in yaml config file."
        self.label_list = label_list

    def __call__(self, preds, **kwargs):
        if isinstance(preds, Tensor):
            preds = preds.asnumpy()

        pred_idxs = preds.argmax(axis=1)

        angles, scores = [], []
        for i, idx in enumerate(pred_idxs):
            angles.append(self.label_list[idx])
            scores.append(preds[i, idx])
        decode_preds = {"angles": angles, "scores": scores}

        return decode_preds
