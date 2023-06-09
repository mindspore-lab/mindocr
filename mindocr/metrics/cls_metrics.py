__all__ = ["ClsMetric"]


class ClsMetric(object):
    """Compute the text direction classification accuracy."""

    def __init__(self, label_list=None, **kwargs):
        """
        label_list: Set in yaml config file, map the gts back to original label format (angle).
        """
        assert (
            label_list is not None
        ), "`label_list` should not be None. Please set it in 'metric' section in yaml config file."
        self.label_list = label_list
        self.eps = 1e-5
        self.metric_names = ["acc"]
        self.clear()

    def update(self, *inputs):
        preds, gts = inputs
        preds = preds["angles"]
        if isinstance(gts, list):
            gts = gts[0]
        gts = [self.label_list[i] for i in gts]

        correct_num = 0
        all_num = 0
        for pred, target in zip(preds, gts):
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num

    def eval(self):
        acc = self.correct_num / (self.all_num + self.eps)
        self.clear()
        return {"acc": acc}

    def clear(self):
        self.correct_num = 0
        self.all_num = 0
