import logging
from math import pi
from typing import Tuple, Union

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops

__all__ = ["DBLoss", "PSEDiceLoss", "EASTLoss", "FCELoss"]
_logger = logging.getLogger(__name__)


class DBLoss(nn.LossBase):
    """
    Apply Balanced CrossEntropy Loss on `binary`, MaskL1Loss on `thresh`, DiceLoss on `thresh_binary` and return
    overall weighted loss.

    Args:
        eps: epsilon value to add to the denominator to avoid division by zero. Default: 1e-6.
        bce_scale: scale coefficient for Balanced CrossEntropy Loss. Default: 5
        l1_scale: scale coefficient for MaskL1Loss. Default: 10.
        bce_replace: loss to be used instead of Balanced CrossEntropy. Choices: ['bceloss', 'diceloss'].
                     Default: 'bceloss'.
    """

    def __init__(self, eps: float = 1e-6, bce_scale: int = 5, l1_scale: int = 10, bce_replace: str = "bceloss"):
        super().__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)

        if bce_replace == "bceloss":
            self.bce_loss = BalancedBCELoss(eps=eps)
        elif bce_replace == "diceloss":
            self.bce_loss = DiceLoss(eps=eps)
        else:
            raise ValueError(f"bce_replace should be in ['bceloss', 'diceloss'], but get {bce_replace}")

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(
        self, pred: Union[Tensor, Tuple[Tensor]], gt: Tensor, gt_mask: Tensor, thresh_map: Tensor, thresh_mask: Tensor
    ) -> Tensor:
        """
        Compute overall weighted loss.

        Args:
            pred: Network prediction that consists of
                binary: The text segmentation prediction.
                thresh: The threshold prediction (optional)
                thresh_binary: Value produced by `step_function(binary - thresh)`. (optional)
            gt: Texts binary map.
            gt_mask: Ignore mask. Pixels with values 1 indicate no contribution to loss.
            thresh_map: Threshold map.
            thresh_mask: Mask that indicates regions used in the threshold map loss calculation.
        Return:
            Tensor: weighted loss value.
        """
        if isinstance(pred, ms.Tensor):
            loss = self.bce_loss(pred, gt, gt_mask)
        else:
            binary, thresh, thresh_binary = pred
            bce_loss_output = self.bce_loss(binary, gt, gt_mask)
            l1_loss = self.l1_loss(thresh, thresh_map, thresh_mask)
            dice_loss = self.dice_loss(thresh_binary, gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + self.bce_scale * bce_loss_output

        return loss


class DiceLoss(nn.LossBase):
    """
    Introduced in `"Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations"
    <https://arxiv.org/abs/1905.02244>`_. Dice loss handles well the class imbalance in terms of pixel count for
    foreground and background.

    Args:
        eps: epsilon value to add to the denominator to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
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
    """
    L1 loss for the masked region.

    Args:
        eps: epsilon value to add to the denominator to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self._eps = eps

    def construct(self, pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
        """
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        """
        pred = pred.squeeze(axis=1)
        return ((pred - gt).abs() * mask).sum() / (mask.sum() + self._eps)


class BalancedBCELoss(nn.LossBase):
    """
    Balanced cross entropy loss - number of false positive pixels that affect the loss value is limited by
    the `negative_ratio` to preserve balance between true and false positives.

    Args:
        negative_ratio: number of negative pixels (false positive) selected in ratio with the number of
                        positive pixels (true positive).
        eps: epsilon value to add to the denominator to avoid division by zero. Default: 1e-6.
    """

    def __init__(self, negative_ratio: int = 3, eps: float = 1e-6):
        super().__init__()
        self._negative_ratio = negative_ratio
        self._eps = eps
        self._bce_loss = ops.BinaryCrossEntropy(reduction="none")

    def construct(self, pred: Tensor, gt: Tensor, mask: Tensor) -> Tensor:
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

        return (pos_loss.sum() + neg_loss.sum()) / (
            pos_count.astype(ms.float32).sum() + neg_count.astype(ms.float32).sum() + self._eps
        )


class FCELoss(nn.LossBase):
    """The class for implementing FCENet loss
    FCENet(CVPR2021): Fourier Contour Embedding for Arbitrary-shaped
        Text Detection
    [https://arxiv.org/abs/2104.10442]
    Args:
        fourier_degree (int) : The maximum Fourier transform degree k.
        num_sample (int) : The sampling points number of regression
            loss. If it is too small, fcenet tends to be overfitting.
        ohem_ratio (float): the negative/positive ratio in OHEM.
    """

    def __init__(self, fourier_degree, num_sample, ohem_ratio=3.0):
        super().__init__()
        self.fourier_degree = fourier_degree
        self.num_sample = num_sample
        self.ohem_ratio = ohem_ratio
        self.sort_descending = ms.ops.Sort(descending=True)
        # equation = "ak, kn-> an"
        # self.einsum = ops.Einsum(equation)
        self.threshold0 = ms.Tensor(0.5, ms.float32)
        self.ohem_thresh = ms.Tensor(1.0, ms.float32)
        self.greater = ops.GreaterEqual()
        self.eps = ms.Tensor(1e-8, ms.float32)

    def construct(self, preds, p3_maps, p4_maps, p5_maps):
        preds_p3, preds_p4, preds_p5 = preds
        # assert isinstance(preds, list)
        # assert p3_maps[0].shape[0] == 4 * self.fourier_degree + 5,\
        #     'fourier degree not equal in FCEhead and FCEtarget'

        # device = preds[0][0].device
        # # to tensor
        # gts = [p3_maps, p4_maps, p5_maps]
        # for idx, maps in enumerate(gts):
        #     gts[idx] = torch.from_numpy(np.stack(maps)).float().to(device)

        # losses = multi_apply(self.forward_single, preds, gts)
        losses = [self.forward_single(preds_p3, p3_maps), self.forward_single(preds_p4, p4_maps)]
        if p5_maps is not None:
            losses.append(self.forward_single(preds_p5, p5_maps))

        loss_tr = ms.Tensor(0.0, ms.float32)  # torch.tensor(0., device=device).float()
        loss_tcl = ms.Tensor(0.0, ms.float32)  # torch.tensor(0., device=device).float()
        loss_reg_x = ms.Tensor(0.0, ms.float32)  # torch.tensor(0., device=device).float()
        loss_reg_y = ms.Tensor(0.0, ms.float32)  # torch.tensor(0., device=device).float()

        for i in range(len(losses)):
            loss_tr += losses[i][0]
            loss_tcl += losses[i][1]
            loss_reg_x += losses[i][2]
            loss_reg_y += losses[i][3]

        all_loss = loss_tr + loss_tcl + loss_reg_x + loss_reg_y
        _logger.info(
            f"all_loss = {all_loss},loss_text={loss_tr},loss_center={loss_tcl},loss_reg_x={loss_reg_x},"
            f"loss_reg_y={loss_reg_y}"
        )
        return all_loss

    def mask_fun(self, pred, mask):
        mask = mask.astype("float32") > 0.5
        return pred * mask

    def forward_single(self, pred, gt):
        cls_pred = ops.Transpose()(pred[0], (0, 2, 3, 1))
        reg_pred = ops.Transpose()(pred[1], (0, 2, 3, 1))
        gt = ops.Transpose()(gt, (0, 2, 3, 1))

        k = 2 * self.fourier_degree + 1
        tr_pred = cls_pred[:, :, :, :2].view((-1, 2))
        tcl_pred = cls_pred[:, :, :, 2:].view((-1, 2))
        x_pred = reg_pred[:, :, :, 0:k].view((-1, k))
        y_pred = reg_pred[:, :, :, k : 2 * k].view((-1, k))

        tr_mask = gt[:, :, :, :1].view((-1))
        tcl_mask = gt[:, :, :, 1:2].view((-1))

        train_mask = gt[:, :, :, 2:3].view((-1))
        x_map = gt[:, :, :, 3 : 3 + k].view((-1, k))
        y_map = gt[:, :, :, 3 + k :].view((-1, k))

        tr_train_mask = train_mask * tr_mask
        # tr loss
        loss_tr = self.ohem(tr_pred, tr_mask.astype(ms.int32), train_mask.astype(ms.int32))

        # tcl loss
        loss_tcl = ms.Tensor(0.0, ms.float32)  # torch.tensor(0.).float().to(device)
        tr_neg_mask = 1 - tr_train_mask

        # print(int(tr_train_mask.sum().item(0)))
        if ops.stop_gradient(tr_train_mask.sum()) > 0:
            loss_tcl_none = ops.cross_entropy(tcl_pred, tcl_mask.astype(ms.int32), reduction="none")
            loss_tcl_pos = (
                loss_tcl_none * self.greater(tr_train_mask, self.threshold0).astype("float32")
            ).sum() / self.greater(tr_train_mask, self.threshold0).astype("float32").sum()

            loss_tcl_neg = (
                loss_tcl_none * self.greater(tr_neg_mask, self.threshold0).astype("float32")
            ).sum() / self.greater(tr_neg_mask, self.threshold0).astype("float32").sum()

            loss_tcl = loss_tcl_pos + 0.5 * loss_tcl_neg

        # regression loss
        loss_reg_x = ms.Tensor(0.0, ms.float32)  # torch.tensor(0.).float().to(device)
        loss_reg_y = ms.Tensor(0.0, ms.float32)  # torch.tensor(0.).float().to(device)

        if ops.stop_gradient(tr_train_mask.sum()) > 0:
            weight = (tr_mask.astype("float32") + tcl_mask.astype("float32")) / 2
            weight = weight.view((-1, 1))
            ft_x, ft_y = self.fourier2poly(x_map, y_map)
            ft_x_pre, ft_y_pre = self.fourier2poly(x_pred, y_pred)
            dim = ft_x.shape[1]

            loss_x = ops.smooth_l1_loss(ft_x_pre, ft_x, reduction="none")
            loss_reg_x = weight * loss_x
            loss_reg_x = (
                (loss_reg_x * self.greater(tr_train_mask.view((-1, 1)), self.threshold0).astype("float32")).sum()
                / self.greater(tr_train_mask.view((-1, 1)), self.threshold0).astype("float32").sum()
                / dim
            )

            loss_y = ops.smooth_l1_loss(ft_y_pre, ft_y, reduction="none")
            loss_reg_y = weight * loss_y
            loss_reg_y = (
                (loss_reg_y * self.greater(tr_train_mask.view((-1, 1)), self.threshold0).astype("float32")).sum()
                / self.greater(tr_train_mask.view((-1, 1)), self.threshold0).astype("float32").sum()
                / dim
            )

        return loss_tr, loss_tcl, loss_reg_x, loss_reg_y

    def _filter_negatives(self, loss: Tensor, n_neg: Tensor) -> Tensor:
        negative_value, _ = self.sort_descending(loss)  # Top K
        con_k = negative_value[n_neg - 1]
        return (negative_value >= con_k).astype(negative_value.dtype)

    def ohem(self, predict, target, train_mask):
        pos = ops.stop_gradient((target * train_mask).astype(ms.float32))
        neg = ((1 - target) * train_mask).astype(ms.float32)

        n_pos = ops.stop_gradient(pos.sum())
        n_neg = ops.select(
            n_pos < self.ohem_thresh,
            ms.Tensor(100, ms.int32),
            ops.minimum(neg.sum(), self.ohem_ratio * n_pos).astype(ms.int32),
        )
        n_neg = ops.stop_gradient(n_neg)

        loss = ops.cross_entropy(predict, target, reduction="none")
        loss_pos = (loss * pos).sum()

        neg = ops.select(neg.sum() > n_neg, self._filter_negatives(loss, n_neg), neg)
        neg = ops.stop_gradient(neg)
        loss_neg = loss * neg

        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg.astype(ms.float32) + 1e-7)

    def fourier2poly(self, real_maps, imag_maps):
        """Transform Fourier coefficient maps to polygon maps.
        Args:
            real_maps (tensor): A map composed of the real parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
            imag_maps (tensor):A map composed of the imag parts of the
                Fourier coefficients, whose shape is (-1, 2k+1)
        Returns
            x_maps (tensor): A map composed of the x value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
            y_maps (tensor): A map composed of the y value of the polygon
                represented by n sample points (xn, yn), whose shape is (-1, n)
        """

        # device = real_maps.device

        # k_vect = torch.arange(
        #     -self.fourier_degree,
        #     self.fourier_degree + 1,
        #     dtype=torch.float,
        #     device=device).view(-1, 1)
        # i_vect = torch.arange(
        #     0, self.num_sample, dtype=torch.float, device=device).view(1, -1)
        k_vect = ms.ops.arange(-self.fourier_degree, self.fourier_degree + 1, dtype=ms.float32).view((-1, 1))
        i_vect = ms.ops.arange(0, self.num_sample, dtype=ms.float32).view((1, -1))

        transform_matrix = 2 * pi / self.num_sample * ops.matmul(k_vect, i_vect)

        # x1 = self.einsum((real_maps,ops.cos(transform_matrix)))
        # x2 = self.einsum((imag_maps,ops.sin(transform_matrix)))
        # y1 = self.einsum((real_maps,ops.sin(transform_matrix)))
        # y2 = self.einsum((imag_maps,ops.cos(transform_matrix)))

        x1 = ops.matmul(real_maps, ops.cos(transform_matrix))
        x2 = ops.matmul(imag_maps, ops.sin(transform_matrix))
        y1 = ops.matmul(real_maps, ops.sin(transform_matrix))
        y2 = ops.matmul(imag_maps, ops.cos(transform_matrix))

        x_maps = x1 - x2
        y_maps = y1 + y2

        return x_maps, y_maps


class PSEDiceLoss(nn.Cell):
    """
    PSE Dice Loss module for text detection.

    This module calculates the Dice loss between the predicted binary segmentation map and the ground truth map.

    Args:
        alpha (float): The weight for text loss. Default is 0.7.
        ohem_ratio (int): The ratio for hard negative example mining. Default is 3.

    Returns:
        Tensor: The computed loss value.
    """

    def __init__(self, alpha=0.7, ohem_ratio=3):
        super().__init__()
        self.threshold0 = Tensor(0.5, mstype.float32)
        self.zero_float32 = Tensor(0.0, mstype.float32)
        self.alpha = alpha
        self.ohem_ratio = ohem_ratio
        self.negative_one_int32 = Tensor(-1, mstype.int32)
        self.concat = ops.Concat()
        self.less_equal = ops.LessEqual()
        self.greater = ops.Greater()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_sum_keep_dims = ops.ReduceSum(keep_dims=True)
        self.reduce_mean = ops.ReduceMean()
        self.reduce_min = ops.ReduceMin()
        self.cast = ops.Cast()
        self.minimum = ops.Minimum()
        self.expand_dims = ops.ExpandDims()
        self.select = ops.Select()
        self.fill = ops.Fill()
        self.topk = ops.TopK(sorted=True)
        self.shape = ops.Shape()
        self.sigmoid = ops.Sigmoid()
        self.reshape = ops.Reshape()
        self.slice = ops.Slice()
        self.logical_and = ops.LogicalAnd()
        self.logical_or = ops.LogicalOr()
        self.equal = ops.Equal()
        self.zeros_like = ops.ZerosLike()
        self.add = ops.Add()
        self.gather = ops.Gather()
        self.upsample = nn.ResizeBilinear()

    def ohem_batch(self, scores, gt_texts, training_masks):
        """
        Perform online hard example mining (OHEM) for a batch of scores, ground truth texts, and training masks.

        Args:
            scores (Tensor): The predicted scores of shape [N * H * W].
            gt_texts (Tensor): The ground truth texts of shape [N * H * W].
            training_masks (Tensor): The training masks of shape [N * H * W].

        Returns:
            Tensor: The selected masks of shape [N * H * W].
        """
        batch_size = scores.shape[0]
        h, w = scores.shape[1:]
        selected_masks = ()
        for i in range(batch_size):
            score = self.slice(scores, (i, 0, 0), (1, h, w))
            score = self.reshape(score, (h, w))

            gt_text = self.slice(gt_texts, (i, 0, 0), (1, h, w))
            gt_text = self.reshape(gt_text, (h, w))

            training_mask = self.slice(training_masks, (i, 0, 0), (1, h, w))
            training_mask = self.reshape(training_mask, (h, w))

            selected_mask = self.ohem_single(score, gt_text, training_mask)
            selected_masks = selected_masks + (selected_mask,)

        selected_masks = self.concat(selected_masks)
        return selected_masks

    def ohem_single(self, score, gt_text, training_mask):
        h, w = score.shape[0:2]
        k = int(h * w)
        pos_num = self.logical_and(self.greater(gt_text, self.threshold0), self.greater(training_mask, self.threshold0))
        pos_num = self.reduce_sum(self.cast(pos_num, mstype.float32))

        neg_num = self.less_equal(gt_text, self.threshold0)
        neg_num = self.reduce_sum(self.cast(neg_num, mstype.float32))
        neg_num = self.minimum(self.ohem_ratio * pos_num, neg_num)
        neg_num = self.cast(neg_num, mstype.int32)

        neg_num = neg_num + k - 1
        neg_mask = self.less_equal(gt_text, self.threshold0)
        ignore_score = self.fill(mstype.float32, (h, w), -1e3)
        neg_score = self.select(neg_mask, score, ignore_score)
        neg_score = self.reshape(neg_score, (h * w,))

        topk_values, _ = self.topk(neg_score, k)
        threshold = self.gather(topk_values, neg_num, 0)

        selected_mask = self.logical_and(
            self.logical_or(self.greater(score, threshold), self.greater(gt_text, self.threshold0)),
            self.greater(training_mask, self.threshold0),
        )

        selected_mask = self.cast(selected_mask, mstype.float32)
        selected_mask = self.expand_dims(selected_mask, 0)

        return selected_mask

    def dice_loss(self, input_params, target, mask):
        """
        Compute the dice loss between input parameters, target, and mask.

        Args:
            input_params (Tensor): The input parameters of shape [N, H, W].
            target (Tensor): The target of shape [N, H, W].
            mask (Tensor): The mask of shape [N, H, W].

        Returns:
            Tensor: The dice loss value.
        """
        batch_size = input_params.shape[0]
        input_sigmoid = self.sigmoid(input_params)

        input_reshape = self.reshape(input_sigmoid, (batch_size, -1))
        target = self.reshape(target, (batch_size, -1))
        mask = self.reshape(mask, (batch_size, -1))

        input_mask = input_reshape * mask
        target = target * mask

        a = self.reduce_sum(input_mask * target, 1)
        b = self.reduce_sum(input_mask * input_mask, 1) + 0.001
        c = self.reduce_sum(target * target, 1) + 0.001
        d = (2 * a) / (b + c)
        dice_loss = self.reduce_mean(d)
        return 1 - dice_loss

    def avg_losses(self, loss_list):
        loss_kernel = loss_list[0]
        for i in range(1, len(loss_list)):
            loss_kernel += loss_list[i]
        loss_kernel = loss_kernel / len(loss_list)
        return loss_kernel

    def construct(self, model_predict, gt_texts, gt_kernels, training_masks):
        """
        Construct the PSE Dice Loss calculation.

        Args:
            model_predict (Tensor): The predicted model outputs of shape [N * 7 * H * W].
            gt_texts (Tensor): The ground truth texts of shape [N * H * W].
            gt_kernels (Tensor): The ground truth kernels of shape [N * 6 * H * W].
            training_masks (Tensor): The training masks of shape [N * H * W].

        Returns:
            Tensor: The computed loss value.
        """
        batch_size = model_predict.shape[0]
        model_predict = self.upsample(model_predict, scale_factor=4)
        h, w = model_predict.shape[2:]
        texts = self.slice(model_predict, (0, 0, 0, 0), (batch_size, 1, h, w))
        texts = self.reshape(texts, (batch_size, h, w))
        selected_masks_text = self.ohem_batch(texts, gt_texts, training_masks)
        loss_text = self.dice_loss(texts, gt_texts, selected_masks_text)
        kernels = []
        loss_kernels = []
        for i in range(1, 7):
            kernel = self.slice(model_predict, (0, i, 0, 0), (batch_size, 1, h, w))
            kernel = self.reshape(kernel, (batch_size, h, w))
            kernels.append(kernel)

        mask0 = self.sigmoid(texts)
        selected_masks_kernels = self.logical_and(
            self.greater(mask0, self.threshold0), self.greater(training_masks, self.threshold0)
        )
        selected_masks_kernels = self.cast(selected_masks_kernels, mstype.float32)

        for i in range(6):
            gt_kernel = self.slice(gt_kernels, (0, i, 0, 0), (batch_size, 1, h, w))
            gt_kernel = self.reshape(gt_kernel, (batch_size, h, w))
            loss_kernel_i = self.dice_loss(kernels[i], gt_kernel, selected_masks_kernels)
            loss_kernels.append(loss_kernel_i)
        loss_kernel = self.avg_losses(loss_kernels)

        loss = self.alpha * loss_text + (1 - self.alpha) * loss_kernel
        return loss


class DiceCoefficient(nn.Cell):
    def __init__(self):
        super(DiceCoefficient, self).__init__()
        self.sum = ops.ReduceSum()
        self.eps = 1e-5

    def construct(self, true_cls, pred_cls):
        intersection = self.sum(true_cls * pred_cls, ())
        union = self.sum(true_cls, ()) + self.sum(pred_cls, ()) + self.eps
        loss = 1.0 - (2 * intersection / union)

        return loss


class EASTLoss(nn.LossBase):
    def __init__(self):
        super(EASTLoss, self).__init__()
        self.split = ops.Split(1, 5)
        self.log = ops.Log()
        self.cos = ops.Cos()
        self.mean = ops.ReduceMean(keep_dims=False)
        self.sum = ops.ReduceSum()
        self.eps = 1e-5
        self.dice = DiceCoefficient()
        self.abs = ops.Abs()

    def construct(self, pred, score_map, geo_map, training_mask):
        pred_score, pred_geo = pred
        ans = self.sum(score_map)
        classification_loss = self.dice(score_map, pred_score * (1 - training_mask))

        # n * 5 * h * w
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = self.split(geo_map)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = self.split(pred_geo)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = self._min(d2_gt, d2_pred) + self._min(d4_gt, d4_pred)
        h_union = self._min(d1_gt, d1_pred) + self._min(d3_gt, d3_pred)

        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -self.log((area_intersect + 1.0) / (area_union + 1.0))  # iou_loss_map
        angle_loss_map = 1 - self.cos(theta_pred - theta_gt)  # angle_loss_map

        angle_loss = self.sum(angle_loss_map * score_map) / ans
        iou_loss = self.sum(iou_loss_map * score_map) / ans
        geo_loss = 10 * angle_loss + iou_loss

        return geo_loss + classification_loss

    def _min(self, a, b):
        return (a + b - self.abs(a - b)) / 2
