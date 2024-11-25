from addict import Dict

from mindspore import nn, ops
from mindspore.common.initializer import HeNormal, HeUniform, Normal

from ...label_assignment import BBoxAssigner
from ...utils.box_utils import delta2bbox
from .mask_head import MaskRCNNConvUpSampleHead
from .roi_extractor import RoIExtractor


class FastRCNNConvFCHead(nn.SequentialCell):

    def __init__(self, in_channel=256, out_channel=1024, resolution=7, conv_dims=[], fc_dims=[1024, 1024]):
        super().__init__()

        for k, conv_dim in enumerate(conv_dims):
            conv = nn.Conv2d(in_channel,
                             conv_dim,
                             kernel_size=3,
                             padding=1,
                             pad_mode='pad',
                             weight_initializer=HeNormal(mode="fan_out", nonlinearity="relu"),
                             has_bias=True,
                             bias_init="zeros")
            self.insert_child_to_cell("conv{}".format(k + 1), conv)
            self.insert_child_to_cell("conv_relu{}".format(k + 1), nn.ReLU())

        self._output_size = in_channel * resolution * resolution
        for k, fc_dim in enumerate(fc_dims):
            if k == 0:
                self.insert_child_to_cell("flatten", nn.Flatten())
            fc = nn.Dense(self._output_size,
                          fc_dim,
                          weight_init=HeUniform(negative_slope=1),
                          has_bias=True,
                          bias_init="zeros")
            self.insert_child_to_cell("fc{}".format(k + 1), fc)
            self.insert_child_to_cell("fc_relu{}".format(k + 1), nn.ReLU())
            self._output_size = fc_dim

    def construct(self, x):
        b, n, c, _, _ = x.shape
        x = x.reshape(b * n, -1)
        for layer in self:
            x = layer(x)
        return x


class FastRCNNOutputLayers(nn.Cell):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    def __init__(self, out_channel, num_classes, cls_agnostic_bbox_reg=True, box_dim=4):
        super().__init__()

        self.num_classes = num_classes

        self.cls_score = nn.Dense(out_channel,
                                  num_classes + 1,
                                  weight_init=Normal(sigma=0.01),
                                  has_bias=True,
                                  bias_init="zeros")
        self.num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Dense(out_channel,
                                  self.num_bbox_reg_classes * box_dim,
                                  weight_init=Normal(sigma=0.001),
                                  has_bias=True,
                                  bias_init="zeros")

    def construct(self, x):
        if x.dim() > 2:
            x = ops.function.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def predict_boxes(self, predictions, proposals):
        if not len(proposals):
            return []
        batch_size, rois_num, _ = proposals.shape
        _, proposal_deltas = predictions
        rois = ops.tile(proposals[:, :, :4].reshape((batch_size, rois_num, 1, 4)), (1, 1, self.num_bbox_reg_classes, 1))
        # rois = rois.reshape((-1, rois.shape[-1]))[:, :4]
        pred_loc = delta2bbox(proposal_deltas.reshape((-1, 4)), rois.reshape((-1, 4)))  # true box xyxy
        pred_loc = pred_loc.reshape((batch_size, rois_num, self.num_bbox_reg_classes * 4))
        return pred_loc

    def predict_probs(self, predictions, proposals):
        batch_size, rois_num, _ = proposals.shape
        scores, _ = predictions
        pred_cls = scores.reshape((batch_size, rois_num, -1))
        pred_cls = ops.softmax(pred_cls, axis=-1)
        return pred_cls


def get_head(cfg):
    if cfg.name == "FastRCNNConvFCHead":
        return FastRCNNConvFCHead(in_channel=cfg.in_channel,
                                  out_channel=cfg.out_channel,
                                  resolution=cfg.pooler_resolution,
                                  conv_dims=cfg.conv_dims,
                                  fc_dims=cfg.fc_dims)
    else:
        raise InterruptedError(f"Not support bbox_head: {cfg.name}")


class CascadeROIHeads(nn.Cell):
    """Cascade RCNN bbox head"""

    def __init__(self, in_channels, with_mask=False, **cfg):
        super(CascadeROIHeads, self).__init__()
        cfg = Dict(cfg)
        cascade_bbox_reg_weights = cfg.roi_box_cascade_head.bbox_reg_weights
        cascade_ious = cfg.roi_box_cascade_head.ious

        self.box_pooler = RoIExtractor(resolution=cfg.roi_box_head.pooler_resolution,
                                       featmap_strides=cfg.roi_extractor.featmap_strides,
                                       pooler_sampling_ratio=cfg.roi_box_head.pooler_sampling_ratio,
                                       pooler_type=cfg.roi_box_head.pooler_type)

        self.box_in_features = cfg.in_features
        self.num_classes = cfg.num_classes
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.loc_loss = nn.SmoothL1Loss(reduction="none")
        self.with_mask = with_mask

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = get_head(cfg.roi_box_head)
            box_heads.append(box_head)
            box_predictors.append(FastRCNNOutputLayers(cfg.roi_box_head.out_channel, self.num_classes,
                                                       cfg.roi_box_head.cls_agnostic_bbox_reg))
            proposal_matchers.append(BBoxAssigner(
                rois_per_batch=cfg.bbox_assigner.rois_per_batch,
                bg_thresh=cfg.bbox_assigner.bg_thresh,
                fg_thresh=cfg.bbox_assigner.fg_thresh,
                fg_fraction=cfg.bbox_assigner.fg_fraction,
                num_classes=cfg.num_classes,
                with_mask=with_mask
            ))

        self.box_head = nn.CellList(box_heads)
        self.box_predictor = nn.CellList(box_predictors)
        self.proposal_matchers = nn.CellList(proposal_matchers)

        self.num_cascade_stages = len(box_heads)

        if cfg.mask_on:
            self.mask_head = MaskRCNNConvUpSampleHead(in_channels=cfg.roi_mask_head.in_channel,
                                                      num_classes=self.num_classes,
                                                      conv_dims=cfg.roi_mask_head.conv_dims)
            self.mask_pooler = RoIExtractor(resolution=cfg.roi_mask_head.pooler_resolution,
                                            featmap_strides=cfg.roi_extractor.featmap_strides,
                                            pooler_sampling_ratio=cfg.roi_mask_head.pooler_sampling_ratio,
                                            pooler_type=cfg.roi_mask_head.pooler_type)

    def construct(self, feats, rois, rois_mask, gts, gt_masks=None):
        """
        feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_mask (Tensor): The number of RoIs in each image
        gts (Tensor): The ground-truth
        """
        pass

    def _run_stage(self, features, proposals, proposals_mask, stage):
        box_features = self.box_pooler(features, proposals, proposals_mask)
        if self.training:
            pass
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features)

    def clip_boxes(self, boxes, im_shape):
        h, w = im_shape
        x1 = ops.clip_by_value(boxes[..., 0], 0, w)
        y1 = ops.clip_by_value(boxes[..., 1], 0, h)
        x2 = ops.clip_by_value(boxes[..., 2], 0, w)
        y2 = ops.clip_by_value(boxes[..., 3], 0, h)
        boxes = ops.stack((x1, y1, x2, y2), -1)
        return boxes

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image = self.clip_boxes(boxes_per_image, image_size)
            if self.training:
                pass
            proposals.append(boxes_per_image)
        return ops.stack(proposals, axis=0)

    def predict(self, features, proposals, proposals_mask, image_sizes):
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
            predictions = self._run_stage(features, proposals, proposals_mask, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

        # Average the scores across heads
        scores = [
            sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
            for scores_per_image in zip(*scores_per_stage)
        ]
        scores = ops.stack(scores, axis=0)
        # Use the boxes of the last head
        predictor, predictions, proposals = head_outputs[-1]
        boxes = predictor.predict_boxes(predictions, proposals)
        boxes = self._create_proposals_from_boxes(boxes, image_sizes)

        res = ops.concat((boxes, scores), axis=-1)
        return res
