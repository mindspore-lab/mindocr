import numpy as np

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn
from mindspore.ops import operations as P

__all__ = ["ABINetLoss"]


class ABINetLoss(nn.Cell):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss()
        self.bce = nn.BCELoss(reduction="mean")
        self.losses = ms.Parameter([])
        self.cast = P.Cast()

    # @property
    def last_losses(self):
        return self.losses

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            concat_op = ms.ops.Concat(axis=0)
            if isinstance(items[0], ms.Tensor):
                return concat_op(items)
            else:
                return items[0]

        res = []

        for key in all_res[0].keys():
            items = []

            for i in range(3):
                items.append(all_res[i][key])

            res.append(merge(items))

        return res

    def _ce_loss(self, output, loss_args, i, idx=None, record=True):
        pt_logits = 1.0
        weight = 1.0

        if i == 0:
            pt_logits = output[0]

        if i == 1:
            pt_logits = output[1]

        if i == 2:
            pt_logits = output["logits"]

        gt_labels = loss_args[0]
        gt_lengths = loss_args[1]
        label_for_mask = loss_args[2]
        assert pt_logits.shape[0] % gt_labels.shape[0] == 0

        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        type_dst = ms.float16
        cast = ms.ops.Cast()
        gt_labels = cast(gt_labels, type_dst)
        gt_lengths = cast(gt_lengths, type_dst)
        pt_logits = cast(pt_logits, type_dst)
        label_for_mask = cast(label_for_mask, type_dst)

        if iter_size > 1:
            gt_labels = msnp.tile(gt_labels, (3, 1, 1))
            gt_lengths = msnp.tile(gt_lengths, 3)

            label_for_mask = msnp.tile(label_for_mask, (3, 1))

        label_for_mask = label_for_mask[:, None]

        loss = self.ce(gt_labels, pt_logits, gt_lengths, label_for_mask) * weight

        return loss

    def construct(self, outputs, label, length, label_for_mask):
        loss_args = [label, length, label_for_mask]
        output_list = []
        for i in range(len(outputs)):
            output_list.append(self._merge_list(outputs[i]))
        outputs = output_list
        loss_one = 0
        loss_all = 0
        for i in range(3):
            loss_one = self._ce_loss(outputs[i], loss_args, i)
            loss_all = loss_one + loss_all
        return loss_all


class SoftCrossEntropyLoss(nn.Cell):
    def __init__(self, reduction="mean"):
        super().__init__()

    def construct(self, gt_labels, pt_logits, gt_lengths, label_for_mask, softmax=True):
        log_softmax = nn.LogSoftmax(axis=-1)
        log = ms.ops.Log()
        concat_op = ms.ops.Concat()
        data_pt_list = []
        mask_list = []
        gt_list = []

        loss = 0
        mean_divide = 0

        for i in range(pt_logits.shape[0]):
            data_length = gt_lengths[i]
            mean_divide = mean_divide + data_length
            mask_pt = label_for_mask[i] > 0

            mask_pt = mask_pt.transpose(1, 0)

            data_pt_list.append(pt_logits[i])
            mask_list.append(mask_pt)
            gt_list.append(gt_labels[i])

        concat_pt_logits = concat_op(data_pt_list)
        concat_mask = concat_op(mask_list)
        concat_gt_labels = concat_op(gt_list)
        concat_mask = concat_mask.astype(ms.float16)
        concat_pt_logits = concat_pt_logits * concat_mask

        if softmax:
            concat_pt_logits = concat_pt_logits.astype(ms.float16)
            log_prob = log_softmax(concat_pt_logits)
        else:
            log_prob = log(concat_pt_logits)

        loss = -(concat_gt_labels * log_prob)
        loss = loss.astype(ms.float16)
        loss = loss * concat_mask
        loss = loss.sum(axis=(-2, -1))
        loss_mean = loss / mean_divide

        return loss_mean


class CharsetMapper(object):
    def __init__(self, max_length=30, null_char="\u2591"):
        self.null_char = null_char
        self.max_length = max_length
        self.label_to_char = self._read_charset()
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)

    def _read_charset(self):
        charset = {}
        charset = {
            0: "░",
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "f",
            7: "g",
            8: "h",
            9: "i",
            10: "j",
            11: "k",
            12: "l",
            13: "m",
            14: "n",
            15: "o",
            16: "p",
            17: "q",
            18: "r",
            19: "s",
            20: "t",
            21: "u",
            22: "v",
            23: "w",
            24: "x",
            25: "y",
            26: "z",
            27: "1",
            28: "2",
            29: "3",
            30: "4",
            31: "5",
            32: "6",
            33: "7",
            34: "8",
            35: "9",
            36: "0",
        }
        self.null_label = 0
        charset[self.null_label] = self.null_char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, "")

    def get_text(self, labels, length=None, padding=True, trim=False):
        """Returns a string corresponding to a sequence of character ids."""
        length = length if length else self.max_length
        labels = [int(a) if isinstance(a, ms.Tensor) else int(a) for a in labels]
        if padding:
            labels = labels + [self.null_label] * (length - len(labels))
        text = "".join([self.label_to_char[label] for label in labels])
        if trim:
            text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """Returns the labels of the corresponding text."""
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length
        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return "0123456789"

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                valid_chars.append(c)
        return "".join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


def onehot(label, depth, device=None):
    label_shape = 26

    onehot_output = np.zeros((label_shape, depth))

    label_expand = np.expand_dims(label, -1)
    label_expand = np.expand_dims(label_expand, -1)
    label_expand_onehot = np.zeros((26, 37))
    a = 0
    for i in label_expand:
        i = int(i)
        label_expand_onehot[a][i] = 1
        a = a + 1

    label_expand_onehot = label_expand_onehot
    onehot_output = label_expand_onehot + onehot_output

    return onehot_output
