import time

import numpy as np

from mindspore import Tensor

__all__ = ["YOLOv8Postprocess", "Layoutlmv3Postprocess"]


class YOLOv8Postprocess(object):
    """return image_id, category_id, bbox and scores."""

    def __init__(
        self,
        conf_thres=0.001,
        iou_thres=0.65,
        conf_free=False,
        multi_label=True,
        time_limit=60.0,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.conf_free = conf_free
        self.multi_label = multi_label
        self.time_limit = time_limit

    def __call__(self, preds, img_shape, meta_info, **kwargs):
        publaynet5class = [2, 1, 5, 4, 3]
        meta_info = [_.numpy() if isinstance(_, Tensor) else _ for _ in meta_info]
        image_ids, ori_shape, hw_scale, pad = meta_info
        preds = preds if isinstance(preds, np.ndarray) else preds.numpy()
        preds = non_max_suppression(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            conf_free=self.conf_free,
            multi_label=True,
            time_limit=self.time_limit,
        )
        # Statistics pred
        result_dicts = list()
        for si, pred in enumerate(preds):
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(
                img_shape[1:], predn[:, :4], ori_shape[si], ratio=hw_scale[si], pad=pad[si]
            )  # native-space pred

            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                result_dicts.append(
                    {
                        "image_id": image_ids[si],
                        "category_id": publaynet5class[int(p[5])],
                        "bbox": [round(x, 3) for x in b],
                        "score": round(p[4], 5),
                    }
                )
        return result_dicts


class Layoutlmv3Postprocess(YOLOv8Postprocess):
    """return image_id, category_id, bbox and scores."""

    def __call__(self, preds, img_shape, meta_info, **kwargs):
        meta_info = [_.numpy() if isinstance(_, Tensor) else _ for _ in meta_info]
        image_ids, ori_shape, hw_scale, pad = meta_info
        preds = preds if isinstance(preds, np.ndarray) else preds.numpy()
        preds = non_max_suppression_for_layoutlmv3(
            preds,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            conf_free=self.conf_free,
            multi_label=True,
            time_limit=self.time_limit,
        )
        # Statistics pred
        result_dicts = list()
        for si, pred in enumerate(preds):
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords_for_layoutlmv3(
                img_shape[-2:], predn[:, :4], ori_shape[si], ratio=hw_scale[si], pad=None
            )  # native-space pred

            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                result_dicts.append(
                    {
                        "image_id": image_ids[si],
                        "category_id": int(p[5]) + 1,
                        "bbox": [round(x, 3) for x in b],
                        "score": round(p[4], 5),
                    }
                )
        return result_dicts


def _nms(xyxys, scores, threshold):
    """Calculate NMS"""
    x1 = xyxys[:, 0]
    y1 = xyxys[:, 1]
    x2 = xyxys[:, 2]
    y2 = xyxys[:, 3]
    scores = scores
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    reserved_boxes = []
    while order.size > 0:
        i = order[0]
        reserved_boxes.append(i)
        max_x1 = np.maximum(x1[i], x1[order[1:]])
        max_y1 = np.maximum(y1[i], y1[order[1:]])
        min_x2 = np.minimum(x2[i], x2[order[1:]])
        min_y2 = np.minimum(y2[i], y2[order[1:]])

        intersect_w = np.maximum(0.0, min_x2 - max_x1)
        intersect_h = np.maximum(0.0, min_y2 - max_y1)
        intersect_area = intersect_w * intersect_h

        ovr = intersect_area / (areas[i] + areas[order[1:]] - intersect_area + 1e-6)
        indexes = np.where(ovr <= threshold)[0]
        order = order[indexes + 1]
    return np.array(reserved_boxes)


def _box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 ([N, 4])
        box2 ([M, 4])
    Returns:
        iou ([N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (
        (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2])).clip(0, None).prod(2)
    )
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    conf_free=False,
    classes=None,
    agnostic=False,
    multi_label=False,
    time_limit=20.0,
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction (ndarray): Prediction. If conf_free is False, prediction on (bs, N, 5+nc) ndarray each point,
            the last dimension meaning [center_x, center_y, width, height, conf, cls0, ...]; If conf_free is True,
            prediction on (bs, N, 4+nc) ndarray each point, the last dimension meaning
            [center_x, center_y, width, height, cls0, ...].
        conf_free (bool): Whether the prediction result include conf.
        time_limit (float): Batch NMS maximum waiting time
        multi_label (bool): Whether to use multiple labels
        agnostic (bool): Whether the NMS is not aware of the category when executed
        classes (list[int]): Filter for a specified category
        iou_thres: (float): IoU threshold for NMS
        conf_thres: (float): Confidence threshold for NMS

    Returns:
         list of detections, on (n,6) ndarray per image, the last dimension meaning [xyxy, conf, cls].
    """

    if not conf_free:
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
    else:
        nc = prediction.shape[2] - 4  # number of classes
        xc = prediction[..., 4:].max(-1) > conf_thres  # candidates
        prediction = np.concatenate(
            (prediction[..., :4], prediction[..., 4:].max(-1, keepdims=True), prediction[..., 4:]), axis=-1
        )

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = time_limit if time_limit > 0 else 1e3  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Scale class with conf
        if not conf_free:
            if nc == 1:
                x[:, 5:] = x[:, 4:5]  # single cls no need to do multiplication.
            else:
                x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = np.concatenate((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[-max_nms:]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = _nms(boxes, scores, iou_thres)  # NMS for per sample

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = _box_iou(boxes[i], boxes) > iou_thres  # iou matrix # (N, M)
            weights = iou * scores[None]  # box weights
            # (N, M) @ (M, 4) / (N, 1)
            x[i, :4] = np.matmul(weights, x[:, :4]) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(
                f"WARNING: Batch NMS time limit {time_limit}s exceeded, this batch "
                f"process {xi + 1}/{prediction.shape[0]} sample."
            )
            break  # time limit exceeded

    return output


def non_max_suppression_for_layoutlmv3(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    conf_free=False,
    classes=None,
    agnostic=True,
    multi_label=False,
    time_limit=20.0,
):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Args:
        prediction (ndarray): Prediction. If conf_free is False, prediction on (bs, N, 5+nc) ndarray each point,
            the last dimension meaning [center_x, center_y, width, height, conf, cls0, ...]; If conf_free is True,
            prediction on (bs, N, 4+nc) ndarray each point, the last dimension meaning
            [center_x, center_y, width, height, cls0, ...].
        conf_free (bool): Whether the prediction result include conf.
        time_limit (float): Batch NMS maximum waiting time
        multi_label (bool): Whether to use multiple labels
        agnostic (bool): Whether the NMS is not aware of the category when executed
        classes (list[int]): Filter for a specified category
        iou_thres: (float): IoU threshold for NMS
        conf_thres: (float): Confidence threshold for NMS

    Returns:
         list of detections, on (n,6) ndarray per image, the last dimension meaning [xyxy, conf, cls].
    """

    if not conf_free:
        nc = prediction.shape[2] - 5  # number of classes
    else:
        nc = prediction.shape[2] - 4  # number of classes
        prediction = np.concatenate(
            (prediction[..., :4], prediction[..., 4:].max(-1, keepdims=True), prediction[..., 4:]), axis=-1
        )

    # Settings
    max_wh = 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = time_limit if time_limit > 0 else 1e3  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # If none remain process next image
        if not x.shape[0]:
            continue

        box = x[:, :4]

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 4:-1] > conf_thres).nonzero()
            x = np.concatenate((box[i], x[i, j + 4, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf, j = x[:, 4:-1].max(1, keepdim=True)
            x = np.concatenate((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[-max_nms:]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = _nms(boxes, scores, iou_thres)  # NMS for per sample

        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = _box_iou(boxes[i], boxes) > iou_thres  # iou matrix # (N, M)
            weights = iou * scores[None]  # box weights
            # (N, M) @ (M, 4) / (N, 1)
            x[i, :4] = np.matmul(weights, x[:, :4]) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(
                f"WARNING: Batch NMS time limit {time_limit}s exceeded, this batch "
                f"process {xi + 1}/{prediction.shape[0]} sample."
            )
            break  # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape, ratio=None, pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape

    if ratio is None:  # calculate from img0_shape
        ratio = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # ratio  = old / new
    else:
        ratio = ratio[0]

    if pad is None:
        padh, padw = (img1_shape[0] - img0_shape[0] * ratio) / 2, (img1_shape[1] - img0_shape[1] * ratio) / 2
    else:
        padh, padw = pad[:]

    coords[:, [0, 2]] -= padw  # x padding
    coords[:, [1, 3]] -= padh  # y padding
    coords[:, [0, 2]] /= ratio  # x rescale
    coords[:, [1, 3]] /= ratio  # y rescale
    coords = _clip_coords(coords, img0_shape)
    return coords


def scale_coords_for_layoutlmv3(img1_shape, coords, img0_shape, ratio=None, pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape

    coords[:, [0, 2]] /= ratio[1]  # x rescale
    coords[:, [1, 3]] /= ratio[0]  # y rescale
    coords = _clip_coords(coords, img0_shape)
    return coords


def _clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0] = boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, img_shape[0])  # y2
    return boxes
