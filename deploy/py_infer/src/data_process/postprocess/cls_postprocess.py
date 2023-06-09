class ClsPostprocess(object):
    """Convert between text-label and text-index"""

    def __init__(self, label_list=None):
        super(ClsPostprocess, self).__init__()
        self.label_list = label_list

    def __call__(self, preds):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        label_list = self.label_list
        if label_list is None:
            label_list = {idx: idx for idx in range(preds.shape[-1])}

        pred_idxs = preds.argmax(axis=-1)
        decode_out = [(label_list[idx], float(preds[i, idx])) for i, idx in enumerate(pred_idxs)]

        return decode_out
