import mindspore as ms
from mindspore import Tensor, nn, ops


class CTCLossForSVTR(nn.Cell):
    def __init__(self, use_focal_loss=False, **kwargs):
        super(CTCLossForSVTR, self).__init__()
        self.softmax = ops.LogSoftmax(axis=-1)
        self.loss_func = nn.CTCLoss(blank=0, reduction="none")
        self.use_focal_loss = use_focal_loss

    def construct(self, predicts, batch):
        if isinstance(predicts, (list, tuple)):
            predicts = predicts[-1]
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = Tensor([N] * B, dtype=ms.int32)
        labels = batch[0].astype(ms.int32)
        label_lengths = batch[2].astype(ms.int32)
        predicts = self.softmax(predicts)
        loss = self.loss_func(predicts, labels, preds_lengths, label_lengths)
        if self.use_focal_loss:
            weight = ops.exp(-loss)
            weight = ops.subtract(Tensor([1.0]), weight)
            weight = ops.square(weight)
            loss = ops.multiply(loss, weight)
        loss = loss.mean()
        return loss


class SARLoss(nn.Cell):
    def __init__(self, **kwargs):
        super(SARLoss, self).__init__()
        ignore_index = kwargs.get("ignore_index", 6626)  # 6626
        self.loss_func = nn.loss.CrossEntropyLoss(reduction="mean", ignore_index=ignore_index)

    def construct(self, predicts, batch):
        predict = predicts[:, :-1, :]  # ignore last index of outputs to be in same seq_len with targets
        label = batch[1][:, 1:]  # ignore first index of target in loss calculation
        num_classes = predict.shape[2]
        assert (
            len(label.shape) == len(list(predict.shape)) - 1
        ), "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = ops.reshape(predict, [-1, num_classes])
        targets = ops.reshape(label, [-1])
        loss = self.loss_func(inputs, targets)
        return loss


class MultiLoss(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_funcs = {}
        self.loss_list = kwargs.pop("loss_config_list")
        self.weight_1 = kwargs.get("weight_1", 1.0)
        self.weight_2 = kwargs.get("weight_2", 1.0)
        for loss_info in self.loss_list:
            for name, param in loss_info.items():
                if param is not None:
                    kwargs.update(param)
                loss = eval(name)(**kwargs)
                self.loss_funcs[name] = loss

    def construct(self, predicts, label_ctc, label_sar, length, valid_ratio):
        predicts_ctc, predicts_sar = predicts[0], predicts[1]
        batch = (
            label_ctc,
            label_sar,
            length,
            valid_ratio,
        )
        total_loss = 0.0
        # batch: [image, label_ctc, label_sar, length, valid_ratio]
        for name, loss_func in self.loss_funcs.items():
            if name == "CTCLossForSVTR":
                loss = loss_func(predicts_ctc, batch) * self.weight_1
            elif name == "SARLoss":
                loss = loss_func(predicts_sar, batch) * self.weight_2
            else:
                raise NotImplementedError("{} is not supported in MultiLoss yet".format(name))
            total_loss += loss
        return total_loss
