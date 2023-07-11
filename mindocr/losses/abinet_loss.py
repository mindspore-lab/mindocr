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
        self.cast = P.Cast()

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

        concat_pt_logits = ms.ops.concat(data_pt_list)
        concat_mask = ms.ops.concat(mask_list)
        concat_gt_labels = ms.ops.concat(gt_list)
        concat_mask = concat_mask.astype(ms.float16)
        concat_pt_logits = concat_pt_logits * concat_mask

        if softmax:
            concat_pt_logits = concat_pt_logits.astype(ms.float16)
            log_prob = ms.ops.log_softmax(concat_pt_logits)
        else:
            log_prob = ms.ops.log(concat_pt_logits)

        loss = -(concat_gt_labels * log_prob)
        loss = loss.astype(ms.float16)
        loss = loss * concat_mask
        loss = loss.sum(axis=(-2, -1))
        loss_mean = loss / mean_divide

        return loss_mean
