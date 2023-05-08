from typing import Tuple, Union
from mindspore import nn, ops
import mindspore as ms
from mindspore import Tensor
import mindspore.numpy as mnp

__all__ = ['L1BalancedCELoss', 'EastLoss']


class L1BalancedCELoss(nn.LossBase):
    """
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    """
    def __init__(self, eps=1e-6, bce_scale=5, l1_scale=10, bce_replace="bceloss"):
        super().__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()

        if bce_replace == "bceloss":
            self.bce_loss = BalancedBCELoss()
        elif bce_replace == "diceloss":
            self.bce_loss = DiceLoss()
        else:
            raise ValueError(f"bce_replace should be in ['bceloss', 'diceloss'], but get {bce_replace}")

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred: Union[Tensor, Tuple[Tensor]], gt: Tensor, gt_mask: Tensor, thresh_map: Tensor, thresh_mask: Tensor):
        """
        Compute dbnet loss
        Args:
            pred (Tuple[Tensor]): network prediction consists of
                binary: The text segmentation prediction.
                thresh: The threshold prediction (optional)
                thresh_binary: Value produced by `step_function(binary - thresh)`. (optional)
            gt (Tensor): Text regions bitmap gt.
            mask (Tensor): Ignore mask, pexels where value is 1 indicates no contribution to loss.
            thresh_mask (Tensor): Mask indicates regions cared by thresh supervision.
            thresh_map (Tensor): Threshold gt.
        Return:
            loss value (Tensor)
        """
        if isinstance(pred, ms.Tensor):
            loss = self.bce_loss(pred, gt, gt_mask)
        else:
            binary, thresh, thresh_binary = pred
            bce_loss_output = self.bce_loss(binary, gt, gt_mask)
            l1_loss = self.l1_loss(thresh, thresh_map, thresh_mask)
            dice_loss = self.dice_loss(thresh_binary, gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss_output

        '''
        if isinstance(pred, tuple):
            binary, thresh, thresh_binary = pred
        else:
            binary = pred

        bce_loss_output = self.bce_loss(binary, gt, gt_mask)

        if isinstance(pred, tuple):
            l1_loss = self.l1_loss(thresh, thresh_map, thresh_mask)
            dice_loss = self.dice_loss(thresh_binary, gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss_output
        else:
            loss = bce_loss_output
        '''
        return loss


class DiceLoss(nn.LossBase):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred, gt, mask):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
              the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        """
        pred = pred.squeeze(axis=1) * mask
        gt = gt.squeeze(axis=1) * mask

        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() + self._eps
        return 1 - 2.0 * intersection / union


class MaskL1Loss(nn.LossBase):
    def __init__(self, eps=1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred, gt, mask):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        pred = pred.squeeze(axis=1)
        return ((pred - gt).abs() * mask).sum() / (mask.sum() + self._eps)


class BalancedBCELoss(nn.LossBase):
    """Balanced cross entropy loss."""
    def __init__(self, negative_ratio=3, eps=1e-6):
        super().__init__()
        self._negative_ratio = negative_ratio
        self._eps = eps
        self._bce_loss = ops.BinaryCrossEntropy(reduction='none')

    def construct(self, pred, gt, mask):
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        pred = pred.squeeze(axis=1)
        gt = gt.squeeze(axis=1)

        positive = gt * mask
        negative = (1 - gt) * mask

        pos_count = positive.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        neg_count = negative.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        neg_count = ops.minimum(neg_count, pos_count * self._negative_ratio).squeeze(axis=(1, 2))

        loss = self._bce_loss(pred, gt, None)

        pos_loss = loss * positive
        neg_loss = (loss * negative).view(loss.shape[0], -1)

        neg_vals, _ = ops.sort(neg_loss)
        neg_index = ops.stack((mnp.arange(loss.shape[0]), neg_vals.shape[1] - neg_count), axis=1)
        min_neg_score = ops.expand_dims(ops.gather_nd(neg_vals, neg_index), axis=1)

        neg_loss_mask = (neg_loss >= min_neg_score).astype(ms.float32)  # filter values less than top k
        neg_loss_mask = ops.stop_gradient(neg_loss_mask)

        neg_loss = neg_loss_mask * neg_loss

        return (pos_loss.sum() + neg_loss.sum()) / \
               (pos_count.astype(ms.float32).sum() + neg_count.astype(ms.float32).sum() + self._eps)


class DiceCoefficient(nn.Cell):
    def __init__(self):
        super(DiceCoefficient, self).__init__()
        self.sum = ops.ReduceSum()
        self.eps = 1e-5

    def construct(self, true_cls, pred_cls):
        intersection = self.sum(true_cls * pred_cls, ())
        union = self.sum(true_cls, ()) + self.sum(pred_cls, ()) + self.eps
        loss = 1. - (2 * intersection / union)

        return loss


class MyMin(nn.Cell):
    def __init__(self):
        super(MyMin, self).__init__()
        self.abs = ops.Abs()

    def construct(self, a, b):
        return (a + b - self.abs(a - b)) / 2


class EastLoss(nn.Cell):
    def __init__(self):
        super(EastLoss, self).__init__()
        self.split = ops.Split(1, 5)
        self.min = MyMin()
        self.log = ops.Log()
        self.cos = ops.Cos()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sum = ops.ReduceSum()
        self.eps = 1e-5
        self.dice = DiceCoefficient()

    def construct(
            self,
            pred,
            score_map,
            geo_map,
            training_mask):
        ans = self.sum(score_map)
        classification_loss = self.dice(
            score_map, pred['score'] * (1 - training_mask))

        # n * 5 * h * w
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = self.split(geo_map)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = self.split(pred['geo'])
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = self.min(d2_gt, d2_pred) + self.min(d4_gt, d4_pred)
        h_union = self.min(d1_gt, d1_pred) + self.min(d3_gt, d3_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -self.log((area_intersect + 1.0) /
                                 (area_union + 1.0))  # iou_loss_map
        angle_loss_map = 1 - self.cos(theta_pred - theta_gt)  # angle_loss_map

        angle_loss = self.sum(angle_loss_map * score_map) / ans
        iou_loss = self.sum(iou_loss_map * score_map) / ans
        geo_loss = 10 * angle_loss + iou_loss

        return geo_loss + classification_loss
