"""
This code is refer from:
https://github.com/JiaquanYe/TableMASTER-mmocr/tree/master/mmocr/models/textrecog/losses
"""

import mindspore as ms
from mindspore import nn, ops


class TableMasterLoss(nn.Cell):
    def __init__(self, ignore_index=-1):
        super(TableMasterLoss, self).__init__()
        self.structure_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")
        self.box_loss = nn.L1Loss(reduction="sum")
        self.eps = 1e-12

    def construct(self, predicts, structure, bboxes, bbox_masks):
        # structure_loss
        structure_probs = predicts[0]
        structure_targets = structure
        structure_targets = structure_targets[:, 1:]
        structure_probs = structure_probs.reshape([-1, structure_probs.shape[-1]])
        structure_targets = structure_targets.reshape([-1])
        structure_targets = ops.cast(structure_targets, ms.int32)
        structure_loss = self.structure_loss(structure_probs, structure_targets)

        structure_loss = structure_loss.mean()

        # box loss
        bboxes_preds = predicts[1]
        bboxes_targets = bboxes[:, 1:, :]
        bbox_masks = bbox_masks[:, 1:]
        # mask empty-bbox or non-bbox structure token's bbox.

        masked_bboxes_preds = bboxes_preds * bbox_masks
        masked_bboxes_targets = bboxes_targets * bbox_masks

        # horizon loss (x and width)
        horizon_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 0::2], masked_bboxes_targets[:, :, 0::2])
        horizon_loss = horizon_sum_loss / (bbox_masks.sum() + self.eps)
        # vertical loss (y and height)
        vertical_sum_loss = self.box_loss(masked_bboxes_preds[:, :, 1::2], masked_bboxes_targets[:, :, 1::2])
        vertical_loss = vertical_sum_loss / (bbox_masks.sum() + self.eps)

        horizon_loss = horizon_loss.mean()
        vertical_loss = vertical_loss.mean()
        all_loss = structure_loss + horizon_loss + vertical_loss
        return all_loss
