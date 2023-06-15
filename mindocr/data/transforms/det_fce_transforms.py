import numpy as np
from PIL import Image, ImageDraw
import cv2
from numpy.fft import fft
from numpy.linalg import norm
import Polygon as plg
import math
import imgaug.augmenters as iaa
import imgaug
import mindspore.dataset.vision as vision
import json
import sys


class DetResizeForTest(object):
    def __init__(self, **kwargs):
        super(DetResizeForTest, self).__init__()
        self.resize_type = 0
        if 'image_shape' in kwargs:
            self.image_shape = kwargs['image_shape']
            self.resize_type = 1
        elif 'limit_side_len' in kwargs:
            self.limit_side_len = kwargs['limit_side_len']
            self.limit_type = kwargs.get('limit_type', 'min')
        elif 'resize_long' in kwargs:
            self.resize_type = 2
            self.resize_long = kwargs.get('resize_long', 960)
        elif 'rescale_img' in kwargs:
            self.resize_type = 3
            self.image_shape = kwargs['rescale_img']
        else:
            self.limit_side_len = 736
            self.limit_type = 'min'

    def __call__(self, data):
        img = data['image']
        src_h, src_w, _ = img.shape
        if self.resize_type == 0:
            # img, shape = self.resize_image_type0(img)
            img, [ratio_h, ratio_w] = self.resize_image_type0(img)
        elif self.resize_type == 2:
            img, [ratio_h, ratio_w] = self.resize_image_type2(img)
        elif self.resize_type == 3:
            img, [ratio_h, ratio_w] = self.resize_image_type3(img)
        else:
            # img, shape = self.resize_image_type1(img)
            img, [ratio_h, ratio_w] = self.resize_image_type1(img)

        data['image'] = img
        data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_type3(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio = min(max(resize_h, resize_w) / max(ori_h, ori_w), min(resize_h, resize_w) / min(ori_h, ori_w))

        resize_h = int(ori_h * ratio + 0.5)
        resize_w = int(ori_w * ratio + 0.5)

        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        return img, [ratio, ratio]

    def resize_image_type1(self, img):
        resize_h, resize_w = self.image_shape
        ori_h, ori_w = img.shape[:2]  # (h, w, c)
        ratio_h = float(resize_h) / ori_h
        ratio_w = float(resize_w) / ori_w
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        # return img, np.array([ori_h, ori_w])
        return img, [ratio_h, ratio_w]

    def resize_image_type0(self, img):
        """
        resize image to a size multiple of 32 which is required by the network
        args:
            img(array): array with shape [h, w, c]
        return(tuple):
            img, (ratio_h, ratio_w)
        """
        limit_side_len = self.limit_side_len
        h, w, c = img.shape

        # limit the max side
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type, image ')
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        try:
            if int(resize_w) <= 0 or int(resize_h) <= 0:
                return None, (None, None)
            img = cv2.resize(img, (int(resize_w), int(resize_h)))
        except:
            print(img.shape, resize_w, resize_h)
            sys.exit(0)
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, [ratio_h, ratio_w]

    def resize_image_type2(self, img):
        h, w, _ = img.shape

        resize_w = w
        resize_h = h

        if resize_h > resize_w:
            ratio = float(self.resize_long) / resize_h
        else:
            ratio = float(self.resize_long) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        img = cv2.resize(img, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return img, [ratio_h, ratio_w]


class FDetLabelEncode(object):
    def __init__(self, min_area=100, **kwargs):
        self.min_area = min_area
        # pass

    def __call__(self, data):
        data['image'] = cv2.imread(data['img_path'], cv2.IMREAD_COLOR)[:, :, ::-1]
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']

            if np.sum(np.array(box, dtype=np.float32), axis=0).min() < 1:
                continue
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        # boxes = np.array(boxes, dtype=np.float32)
        # txt_tags = np.array(txt_tags, dtype=np.bool)
        data['polys'] = np.array(boxes, dtype=np.float32)
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        data['polygons'] = np.array([boxes[i] for i in range(len(txt_tags)) if not txt_tags[i]], dtype=np.float32)
        data['ignore_polygons'] = np.array([boxes[i] for i in range(len(txt_tags)) if txt_tags[i]], dtype=np.float32)

        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None,
             backend=None):
    """Resize image to a given size.

    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
        backend (str | None): The image resize backend type. Options are `cv2`,
            `pillow`, `None`. If backend is None, the global imread_backend
            specified by ``mmcv.use_backend()`` will be used. Default: None.

    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    cv2_interp_codes = {
        'nearest': cv2.INTER_NEAREST,
        'bilinear': cv2.INTER_LINEAR,
        'bicubic': cv2.INTER_CUBIC,
        'area': cv2.INTER_AREA,
        'lanczos': cv2.INTER_LANCZOS4
    }
    h, w = img.shape[:2]
    if backend is None:
        backend = 'cv2'
    if backend not in ['cv2', 'pillow']:
        raise ValueError(f'backend: {backend} is not supported for resize.'
                         f"Supported backends are 'cv2', 'pillow'")

    if backend == 'pillow':
        assert img.dtype == np.uint8, 'Pillow backend only support uint8 type'
        pil_image = Image.fromarray(img)
        pil_image = pil_image.resize(size, pillow_interp_codes[interpolation])
        resized_img = np.array(pil_image)
    else:
        resized_img = cv2.resize(
            img, size, dst=out, interpolation=cv2_interp_codes[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


class RandomScaling:

    def __init__(self, size=800, scale=(3. / 4, 5. / 2), **kwargs):
        """Random scale the image while keeping aspect.

        Args:
            size (int) : Base size before scaling.
            scale (tuple(float)) : The range of scaling.
        """
        assert isinstance(size, int)
        assert isinstance(scale, float) or isinstance(scale, tuple) or isinstance(scale, list)
        self.size = size
        self.scale = scale if (isinstance(scale, tuple) or isinstance(scale, list)) \
            else (1 - scale, 1 + scale)

    def __call__(self, data):
        image = data['image']
        h, w, _ = image.shape

        aspect_ratio = np.random.uniform(min(self.scale), max(self.scale))
        scales = self.size * 1.0 / max(h, w) * aspect_ratio
        scales = np.array([scales, scales])
        out_size = (int(h * scales[1]), int(w * scales[0]))
        image = imresize(image, out_size[::-1])

        data['image'] = image

        for key in ['polygons', 'ignore_polygons']:
            # for key in ['polys']:

            if len(data[key]) == 0:
                continue

            text_polys = np.array(data[key], dtype=np.float32)
            text_polys[:, :, 0::2] = text_polys[:, :, 0::2] * scales[1]
            text_polys[:, :, 1::2] = text_polys[:, :, 1::2] * scales[0]
            data[key] = text_polys
        return data


def poly_intersection(poly_det, poly_gt):
    """Calculate the intersection area between two polygon.

    Args:
        poly_det (Polygon): A polygon predicted by detector.
        poly_gt (Polygon): A gt polygon.

    Returns:
        intersection_area (float): The intersection area between two polygons.
    """
    assert isinstance(poly_det, plg.Polygon)
    assert isinstance(poly_gt, plg.Polygon)

    poly_inter = poly_det & poly_gt
    if len(poly_inter) == 0:
        return 0, poly_inter
    return poly_inter.area(), poly_inter


class RandomCropFlip:

    def __init__(self,
                 pad_ratio=0.1,
                 crop_ratio=0.5,
                 iter_num=1,
                 min_area_ratio=0.2,
                 **kwargs
                 ):
        """Random crop and flip a patch of the image.

        Args:
            crop_ratio (float): The ratio of cropping.
            iter_num (int): Number of operations.
            min_area_ratio (float): Minimal area ratio between cropped patch
                and original image.
        """
        assert isinstance(crop_ratio, float)
        assert isinstance(iter_num, int)
        assert isinstance(min_area_ratio, float)

        self.pad_ratio = pad_ratio
        self.epsilon = 1e-2
        self.crop_ratio = crop_ratio
        self.iter_num = iter_num
        self.min_area_ratio = min_area_ratio

    def __call__(self, results):
        for i in range(self.iter_num):
            results = self.random_crop_flip(results)

        return results

    def random_crop_flip(self, results):
        image = results['image']
        polygons = results['polygons']
        ignore_polygons = results['ignore_polygons']

        all_polygons = list(polygons) + list(ignore_polygons)

        if len(polygons) == 0:
            return results

        if np.random.random() >= self.crop_ratio:
            return results

        h, w, _ = image.shape
        area = h * w
        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        h_axis, w_axis = self.generate_crop_target(image, all_polygons, pad_h,
                                                   pad_w)
        if len(h_axis) == 0 or len(w_axis) == 0:
            return results

        attempt = 0
        while attempt < 10:
            attempt += 1
            polys_keep = []
            polys_new = []
            ign_polys_keep = []
            ign_polys_new = []
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx) - pad_w
            xmax = np.max(xx) - pad_w
            xmin = np.clip(xmin, 0, w - 1)
            xmax = np.clip(xmax, 0, w - 1)
            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy) - pad_h
            ymax = np.max(yy) - pad_h
            ymin = np.clip(ymin, 0, h - 1)
            ymax = np.clip(ymax, 0, h - 1)
            if (xmax - xmin) * (ymax - ymin) < area * self.min_area_ratio:
                # area too small
                continue

            pts = np.stack([[xmin, xmax, xmax, xmin],
                            [ymin, ymin, ymax, ymax]]).T.astype(np.int32)
            pp = plg.Polygon(pts)
            fail_flag = False
            for polygon in polygons:
                ppi = plg.Polygon(polygon.reshape(-1, 2))
                ppiou, _ = poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area())) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area())) < self.epsilon:
                    polys_new.append(polygon)
                else:
                    polys_keep.append(polygon)

            for polygon in ignore_polygons:
                ppi = plg.Polygon(polygon.reshape(-1, 2))
                ppiou, _ = poly_intersection(ppi, pp)
                if np.abs(ppiou - float(ppi.area())) > self.epsilon and \
                        np.abs(ppiou) > self.epsilon:
                    fail_flag = True
                    break
                elif np.abs(ppiou - float(ppi.area())) < self.epsilon:
                    ign_polys_new.append(polygon)
                else:
                    ign_polys_keep.append(polygon)

            if fail_flag:
                continue
            else:
                break

        cropped = image[ymin:ymax, xmin:xmax, :]
        select_type = np.random.randint(3)
        if select_type == 0:
            img = np.ascontiguousarray(cropped[:, ::-1])
        elif select_type == 1:
            img = np.ascontiguousarray(cropped[::-1, :])
        else:
            img = np.ascontiguousarray(cropped[::-1, ::-1])
        image[ymin:ymax, xmin:xmax, :] = img
        results['image'] = image

        if len(polys_new) + len(ign_polys_new) != 0:
            height, width, _ = cropped.shape
            if select_type == 0:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    polys_new[idx] = poly
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    ign_polys_new[idx] = poly
            elif select_type == 1:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    ign_polys_new[idx] = poly
            else:
                for idx, polygon in enumerate(polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    polys_new[idx] = poly
                for idx, polygon in enumerate(ign_polys_new):
                    poly = polygon.reshape(-1, 2)
                    poly[:, 0] = width - poly[:, 0] + 2 * xmin
                    poly[:, 1] = height - poly[:, 1] + 2 * ymin
                    ign_polys_new[idx] = poly
            polygons = polys_keep + polys_new
            ignore_polygons = ign_polys_keep + ign_polys_new
            results['polygons'] = polygons
            results['ignore_polygons'] = ignore_polygons

        return results

    def generate_crop_target(self, image, all_polys, pad_h, pad_w):
        """Generate crop target and make sure not to crop the polygon
        instances.

        Args:
            image (ndarray): The image waited to be crop.
            all_polys (list[list[ndarray]]): All polygons including ground
                truth polygons and ground truth ignored polygons.
            pad_h (int): Padding length of height.
            pad_w (int): Padding length of width.
        Returns:
            h_axis (ndarray): Vertical cropping range.
            w_axis (ndarray): Horizontal cropping range.
        """
        h, w, _ = image.shape
        h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
        w_array = np.zeros((w + pad_w * 2), dtype=np.int32)

        text_polys = []
        for polygon in all_polys:
            rect = cv2.minAreaRect(polygon.astype(np.int32).reshape(-1, 2))
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            text_polys.append([box[0], box[1], box[2], box[3]])

        polys = np.array(text_polys, dtype=np.int32)
        for poly in polys:
            poly = np.round(poly, decimals=0).astype(np.int32)
            minx = np.min(poly[:, 0])
            maxx = np.max(poly[:, 0])
            w_array[minx + pad_w:maxx + pad_w] = 1
            miny = np.min(poly[:, 1])
            maxy = np.max(poly[:, 1])
            h_array[miny + pad_h:maxy + pad_h] = 1

        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]
        return h_axis, w_axis


class RandomCropPolyInstances:  # 会变
    """Randomly crop images and make sure to contain at least one intact
    instance."""

    def __init__(self,
                 crop_ratio=5.0 / 8.0,
                 min_side_ratio=0.4, **kwargs):
        super().__init__()
        self.crop_ratio = crop_ratio
        self.min_side_ratio = min_side_ratio

    def sample_valid_start_end(self, valid_array, min_len, max_start, min_end):

        assert isinstance(min_len, int)
        assert len(valid_array) > min_len

        start_array = valid_array.copy()
        max_start = min(len(start_array) - min_len, max_start)
        start_array[max_start:] = 0
        start_array[0] = 1
        diff_array = np.hstack([0, start_array]) - np.hstack([start_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        start = np.random.randint(region_starts[region_ind],
                                  region_ends[region_ind])

        end_array = valid_array.copy()
        min_end = max(start + min_len, min_end)
        end_array[:min_end] = 0
        end_array[-1] = 1
        diff_array = np.hstack([0, end_array]) - np.hstack([end_array, 0])
        region_starts = np.where(diff_array < 0)[0]
        region_ends = np.where(diff_array > 0)[0]
        region_ind = np.random.randint(0, len(region_starts))
        end = np.random.randint(region_starts[region_ind],
                                region_ends[region_ind])
        return start, end

    def sample_crop_box(self, img_size, results):
        """Generate crop box and make sure not to crop the polygon instances.

        Args:
            img_size (tuple(int)): The image size (h, w).
            results (dict): The results dict.
        """

        assert isinstance(img_size, tuple)
        h, w = img_size[:2]

        key_masks = results['polygons']

        x_valid_array = np.ones(w, dtype=np.int32)
        y_valid_array = np.ones(h, dtype=np.int32)

        selected_mask = key_masks[np.random.randint(0, len(key_masks))]
        selected_mask = selected_mask.reshape((-1, 2)).astype(np.int32)
        max_x_start = max(np.min(selected_mask[:, 0]) - 2, 0)
        min_x_end = min(np.max(selected_mask[:, 0]) + 3, w - 1)
        max_y_start = max(np.min(selected_mask[:, 1]) - 2, 0)
        min_y_end = min(np.max(selected_mask[:, 1]) + 3, h - 1)

        # for key in results.get('mask_fields', []):
        #     if len(results[key].masks) == 0:
        #         continue
        #     masks = results[key].masks
        for key in ['polygons', 'ignore_polygons']:

            if len(results[key]) == 0:
                continue

            key_masks = results[key]
            for mask in key_masks:
                # assert len(mask) == 1
                mask = mask.reshape((-1, 2)).astype(np.int32)
                clip_x = np.clip(mask[:, 0], 0, w - 1)
                clip_y = np.clip(mask[:, 1], 0, h - 1)
                min_x, max_x = np.min(clip_x), np.max(clip_x)
                min_y, max_y = np.min(clip_y), np.max(clip_y)

                x_valid_array[min_x - 2:max_x + 3] = 0
                y_valid_array[min_y - 2:max_y + 3] = 0

        min_w = int(w * self.min_side_ratio)
        min_h = int(h * self.min_side_ratio)

        x1, x2 = self.sample_valid_start_end(x_valid_array, min_w, max_x_start,
                                             min_x_end)
        y1, y2 = self.sample_valid_start_end(y_valid_array, min_h, max_y_start,
                                             min_y_end)

        return np.array([x1, y1, x2, y2])

    def crop_img(self, img, bbox):
        assert img.ndim == 3
        h, w, _ = img.shape
        assert 0 <= bbox[1] < bbox[3] <= h
        assert 0 <= bbox[0] < bbox[2] <= w
        return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def __call__(self, results):
        image = results['image']
        if len(results['polygons']) < 1:
            return results

        if np.random.random_sample() < self.crop_ratio:

            crop_box = self.sample_crop_box(image.shape, results)
            img = self.crop_img(image, crop_box)
            results['image'] = img
            # crop and filter masks
            x1, y1, x2, y2 = crop_box
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)

            for key in ['polygons', 'ignore_polygons']:
                if len(results[key]) == 0:
                    continue
                polygons = np.array(results[key], dtype=np.float32)

                polygons[:, :, 0::2] = polygons[:, :, 0::2] - x1
                polygons[:, :, 1::2] = polygons[:, :, 1::2] - y1

                valid_masks_list = []
                for ind, polygon in enumerate(polygons):
                    if (polygon[:, ::2] >
                        -4).all() and (polygon[:, ::2] < w + 4).all() and (
                            polygon[:, 1::2] > -4).all() and (polygon[:, 1::2] <
                                                              h + 4).all():
                        polygon[:, ::2] = np.clip(polygon[:, ::2], 0, w)
                        polygon[:, 1::2] = np.clip(polygon[:, 1::2], 0, h)
                        valid_masks_list.append(polygon)

                results[key] = np.array(valid_masks_list)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class RandomRotatePolyInstances:

    def __init__(self,
                 rotate_ratio=0.5,
                 max_angle=10,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Randomly rotate images and polygon masks.

        Args:
            rotate_ratio (float): The ratio of samples to operate rotation.
            max_angle (int): The maximum rotation angle.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rotated image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        self.rotate_ratio = rotate_ratio
        self.max_angle = max_angle
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def rotate(self, center, points, theta, center_shift=(0, 0)):
        # rotate points.
        (center_x, center_y) = center
        center_y = -center_y
        x, y = points[:, ::2], points[:, 1::2]
        y = -y

        theta = theta / 180 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        x = (x - center_x)
        y = (y - center_y)

        _x = center_x + x * cos - y * sin + center_shift[0]
        _y = -(center_y + x * sin + y * cos) + center_shift[1]

        points[:, ::2], points[:, 1::2] = _x, _y
        return points

    def cal_canvas_size(self, ori_size, degree):
        assert isinstance(ori_size, tuple)
        angle = degree * math.pi / 180.0
        h, w = ori_size[:2]

        cos = math.cos(angle)
        sin = math.sin(angle)
        canvas_h = int(w * math.fabs(sin) + h * math.fabs(cos))
        canvas_w = int(w * math.fabs(cos) + h * math.fabs(sin))

        canvas_size = (canvas_h, canvas_w)
        return canvas_size

    def sample_angle(self, max_angle):
        angle = np.random.random_sample() * 2 * max_angle - max_angle
        return angle

    def rotate_img(self, img, angle, canvas_size):
        h, w = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotation_matrix[0, 2] += int((canvas_size[1] - w) / 2)
        rotation_matrix[1, 2] += int((canvas_size[0] - h) / 2)

        if self.pad_with_fixed_color:
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                flags=cv2.INTER_NEAREST,
                borderValue=self.pad_value)
        else:
            mask = np.zeros_like(img)
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            img_cut = imresize(img_cut, (canvas_size[1], canvas_size[0]))
            mask = cv2.warpAffine(
                mask,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[1, 1, 1])
            target_img = cv2.warpAffine(
                img,
                rotation_matrix, (canvas_size[1], canvas_size[0]),
                borderValue=[0, 0, 0])
            target_img = target_img + img_cut * mask

        return target_img

    def __call__(self, results):
        if np.random.random_sample() < self.rotate_ratio:
            image = results['image']
            h, w = image.shape[:2]

            angle = self.sample_angle(self.max_angle)
            canvas_size = self.cal_canvas_size((h, w), angle)
            center_shift = (int(
                (canvas_size[1] - w) / 2), int((canvas_size[0] - h) / 2))

            # rotate image
            image = self.rotate_img(image, angle, canvas_size)
            results['image'] = image

            # rotate polygons
            for key in ['polygons', 'ignore_polygons']:
                if len(results[key]) == 0:
                    continue
                polygons = results[key]

                rotated_masks = []
                for mask in polygons:
                    rotated_mask = self.rotate((w / 2, h / 2), mask, angle,
                                               center_shift)
                    rotated_masks.append(rotated_mask)
                results[key] = np.array(rotated_masks)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class SquareResizePad:

    def __init__(self,
                 target_size,
                 pad_ratio=0.6,
                 pad_with_fixed_color=False,
                 pad_value=(0, 0, 0),
                 **kwargs):
        """Resize or pad images to be square shape.

        Args:
            target_size (int): The target size of square shaped image.
            pad_with_fixed_color (bool): The flag for whether to pad rotated
               image with fixed value. If set to False, the rescales image will
               be padded onto cropped image.
            pad_value (tuple(int)): The color value for padding rotated image.
        """
        assert isinstance(target_size, int)
        assert isinstance(pad_ratio, float)
        assert isinstance(pad_with_fixed_color, bool)
        assert isinstance(pad_value, tuple)

        self.target_size = target_size
        self.pad_ratio = pad_ratio
        self.pad_with_fixed_color = pad_with_fixed_color
        self.pad_value = pad_value

    def resize_img(self, img, keep_ratio=True):
        h, w, _ = img.shape
        if keep_ratio:
            t_h = self.target_size if h >= w else int(h * self.target_size / w)
            t_w = self.target_size if h <= w else int(w * self.target_size / h)
        else:
            t_h = t_w = self.target_size
        img = imresize(img, (t_w, t_h))
        return img, (t_h, t_w)

    def square_pad(self, img):
        h, w = img.shape[:2]
        if h == w:
            return img, (0, 0)
        pad_size = max(h, w)
        if self.pad_with_fixed_color:
            expand_img = np.ones((pad_size, pad_size, 3), dtype=np.uint8)
            expand_img[:] = self.pad_value
        else:
            (h_ind, w_ind) = (np.random.randint(0, h * 7 // 8),
                              np.random.randint(0, w * 7 // 8))
            img_cut = img[h_ind:(h_ind + h // 9), w_ind:(w_ind + w // 9)]
            expand_img = imresize(img_cut, (pad_size, pad_size))
        if h > w:
            y0, x0 = 0, (h - w) // 2
        else:
            y0, x0 = (w - h) // 2, 0
        expand_img[y0:y0 + h, x0:x0 + w] = img
        offset = (x0, y0)

        return expand_img, offset

    def square_pad_mask(self, points, offset):
        x0, y0 = offset
        pad_points = points.copy()
        pad_points[::2] = pad_points[::2] + x0
        pad_points[1::2] = pad_points[1::2] + y0
        return pad_points

    def __call__(self, results):
        image = results['image']

        h, w = image.shape[:2]

        if np.random.random_sample() < self.pad_ratio:
            image, out_size = self.resize_img(image, keep_ratio=True)
            image, offset = self.square_pad(image)
        else:
            image, out_size = self.resize_img(image, keep_ratio=False)
            offset = (0, 0)
        # image, out_size = self.resize_img(image, keep_ratio=True)
        # image, offset = self.square_pad(image)
        results['image'] = image
        for key in ['polygons', 'ignore_polygons']:
            if len(results[key]) == 0:
                continue
            polys = np.array(results[key])
            polys[:, :, 0::2] = polys[:, :, 0::2] * out_size[1] / w + offset[0]
            polys[:, :, 1::2] = polys[:, :, 1::2] * out_size[0] / h + offset[1]
            results[key] = polys

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class AugmenterBuilder(object):
    def __init__(self, **kwargs):
        pass

    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        elif isinstance(args, list):
            if root:
                sequence = [self.build(value, root=False) for value in args]
                return iaa.Sequential(sequence)
            else:
                return getattr(iaa, args[0])(
                    *[self.to_tuple_if_list(a) for a in args[1:]])
        elif isinstance(args, dict):
            cls = getattr(iaa, args['type'])
            return cls(**{
                k: self.to_tuple_if_list(v)
                for k, v in args['args'].items()
            })
        else:
            raise RuntimeError('unknown augmenter arg: ' + str(args))

    def to_tuple_if_list(self, obj):
        if isinstance(obj, list):
            return tuple(obj)
        return obj


class Pad(object):
    def __init__(self, size_div=32, **kwargs):
        self.size_div = size_div

    def __call__(self, data):
        img = data['image']
        resize_h2 = max(int(math.ceil(img.shape[0] / 32) * 32), 32)
        resize_w2 = max(int(math.ceil(img.shape[1] / 32) * 32), 32)
        img = cv2.copyMakeBorder(img, 0, resize_h2 - img.shape[0], 0, resize_w2 - img.shape[1], cv2.BORDER_CONSTANT,
                                 value=0)
        data['image'] = img
        return data


class FCENetTargets:
    """Generate the ground truth targets of FCENet: Fourier Contour Embedding
    for Arbitrary-Shaped Text Detection.

    [https://arxiv.org/abs/2104.10442]

    Args:
        fourier_degree (int): The maximum Fourier transform degree k.
        resample_step (float): The step size for resampling the text center
            line (TCL). It's better not to exceed half of the minimum width.
        center_region_shrink_ratio (float): The shrink ratio of text center
            region.
        level_size_divisors (tuple(int)): The downsample ratio on each level.
        level_proportion_range (tuple(tuple(int))): The range of text sizes
            assigned to each level.
    """

    def __init__(self,
                 fourier_degree=5,
                 resample_step=4.0,
                 center_region_shrink_ratio=0.3,
                 level_size_divisors=(8, 16, 32),
                 level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
                 orientation_thr=2.0,
                 **kwargs):

        super().__init__()

        assert isinstance(level_size_divisors, (tuple, list))
        assert isinstance(level_proportion_range, (tuple, list)), type(level_proportion_range)
        assert len(level_size_divisors) == len(level_proportion_range)
        self.fourier_degree = fourier_degree
        self.resample_step = resample_step
        self.center_region_shrink_ratio = center_region_shrink_ratio
        self.level_size_divisors = tuple(level_size_divisors)
        self.level_proportion_range = tuple(level_proportion_range)

        self.orientation_thr = orientation_thr

    def vector_angle(self, vec1, vec2):
        if vec1.ndim > 1:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec1 = vec1 / (norm(vec1, axis=-1) + 1e-8)
        if vec2.ndim > 1:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8).reshape((-1, 1))
        else:
            unit_vec2 = vec2 / (norm(vec2, axis=-1) + 1e-8)
        return np.arccos(
            np.clip(np.sum(unit_vec1 * unit_vec2, axis=-1), -1.0, 1.0))

    def vector_slope(self, vec):
        assert len(vec) == 2
        return abs(vec[1] / (vec[0] + 1e-8))

    def vector_sin(self, vec):
        assert len(vec) == 2
        return vec[1] / (norm(vec) + 1e-8)

    def vector_cos(self, vec):
        assert len(vec) == 2
        return vec[0] / (norm(vec) + 1e-8)

    def resample_line(self, line, n):
        """Resample n points on a line.

        Args:
            line (ndarray): The points composing a line.
            n (int): The resampled points number.

        Returns:
            resampled_line (ndarray): The points composing the resampled line.
        """

        assert line.ndim == 2
        assert line.shape[0] >= 2
        assert line.shape[1] == 2
        assert isinstance(n, int)
        assert n > 0

        length_list = [
            norm(line[i + 1] - line[i]) for i in range(len(line) - 1)
        ]
        total_length = sum(length_list)
        length_cumsum = np.cumsum([0.0] + length_list)
        delta_length = total_length / (float(n) + 1e-8)

        current_edge_ind = 0
        resampled_line = [line[0]]

        for i in range(1, n):
            current_line_len = i * delta_length

            while current_line_len >= length_cumsum[current_edge_ind + 1]:
                current_edge_ind += 1
            current_edge_end_shift = current_line_len - length_cumsum[
                current_edge_ind]
            end_shift_ratio = current_edge_end_shift / length_list[
                current_edge_ind]
            current_point = line[current_edge_ind] + (
                    line[current_edge_ind + 1] -
                    line[current_edge_ind]) * end_shift_ratio
            resampled_line.append(current_point)

        resampled_line.append(line[-1])
        resampled_line = np.array(resampled_line)

        return resampled_line

    def reorder_poly_edge(self, points):
        """Get the respective points composing head edge, tail edge, top
        sideline and bottom sideline.

        Args:
            points (ndarray): The points composing a text polygon.

        Returns:
            head_edge (ndarray): The two points composing the head edge of text
                polygon.
            tail_edge (ndarray): The two points composing the tail edge of text
                polygon.
            top_sideline (ndarray): The points composing top curved sideline of
                text polygon.
            bot_sideline (ndarray): The points composing bottom curved sideline
                of text polygon.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2

        head_inds, tail_inds = self.find_head_tail(points,
                                                   self.orientation_thr)
        head_edge, tail_edge = points[head_inds], points[tail_inds]

        pad_points = np.vstack([points, points])
        if tail_inds[1] < 1:
            tail_inds[1] = len(points)
        sideline1 = pad_points[head_inds[1]:tail_inds[1]]
        sideline2 = pad_points[tail_inds[1]:(head_inds[1] + len(points))]
        sideline_mean_shift = np.mean(
            sideline1, axis=0) - np.mean(
            sideline2, axis=0)

        if sideline_mean_shift[1] > 0:
            top_sideline, bot_sideline = sideline2, sideline1
        else:
            top_sideline, bot_sideline = sideline1, sideline2

        return head_edge, tail_edge, top_sideline, bot_sideline

    def find_head_tail(self, points, orientation_thr):
        """Find the head edge and tail edge of a text polygon.

        Args:
            points (ndarray): The points composing a text polygon.
            orientation_thr (float): The threshold for distinguishing between
                head edge and tail edge among the horizontal and vertical edges
                of a quadrangle.

        Returns:
            head_inds (list): The indexes of two points composing head edge.
            tail_inds (list): The indexes of two points composing tail edge.
        """

        assert points.ndim == 2
        assert points.shape[0] >= 4
        assert points.shape[1] == 2
        assert isinstance(orientation_thr, float)

        if len(points) > 4:
            pad_points = np.vstack([points, points[0]])
            edge_vec = pad_points[1:] - pad_points[:-1]

            theta_sum = []
            adjacent_vec_theta = []
            for i, edge_vec1 in enumerate(edge_vec):
                adjacent_ind = [x % len(edge_vec) for x in [i - 1, i + 1]]
                adjacent_edge_vec = edge_vec[adjacent_ind]
                temp_theta_sum = np.sum(
                    self.vector_angle(edge_vec1, adjacent_edge_vec))
                temp_adjacent_theta = self.vector_angle(
                    adjacent_edge_vec[0], adjacent_edge_vec[1])
                theta_sum.append(temp_theta_sum)
                adjacent_vec_theta.append(temp_adjacent_theta)
            theta_sum_score = np.array(theta_sum) / np.pi
            adjacent_theta_score = np.array(adjacent_vec_theta) / np.pi
            poly_center = np.mean(points, axis=0)
            edge_dist = np.maximum(
                norm(pad_points[1:] - poly_center, axis=-1),
                norm(pad_points[:-1] - poly_center, axis=-1))
            dist_score = edge_dist / np.max(edge_dist)
            position_score = np.zeros(len(edge_vec))
            score = 0.5 * theta_sum_score + 0.15 * adjacent_theta_score
            score += 0.35 * dist_score
            if len(points) % 2 == 0:
                position_score[(len(score) // 2 - 1)] += 1
                position_score[-1] += 1
            score += 0.1 * position_score
            pad_score = np.concatenate([score, score])
            score_matrix = np.zeros((len(score), len(score) - 3))
            x = np.arange(len(score) - 3) / float(len(score) - 4)
            gaussian = 1. / (np.sqrt(2. * np.pi) * 0.5) * np.exp(-np.power(
                (x - 0.5) / 0.5, 2.) / 2)
            gaussian = gaussian / np.max(gaussian)
            for i in range(len(score)):
                score_matrix[i, :] = score[i] + pad_score[
                                                (i + 2):(i + len(score) - 1)] * gaussian * 0.3

            head_start, tail_increment = np.unravel_index(
                score_matrix.argmax(), score_matrix.shape)
            tail_start = (head_start + tail_increment + 2) % len(points)
            head_end = (head_start + 1) % len(points)
            tail_end = (tail_start + 1) % len(points)

            if head_end > tail_end:
                head_start, tail_start = tail_start, head_start
                head_end, tail_end = tail_end, head_end
            head_inds = [head_start, head_end]
            tail_inds = [tail_start, tail_end]
        else:
            if self.vector_slope(points[1] - points[0]) + self.vector_slope(
                    points[3] - points[2]) < self.vector_slope(
                points[2] - points[1]) + self.vector_slope(points[0] -
                                                           points[3]):
                horizontal_edge_inds = [[0, 1], [2, 3]]
                vertical_edge_inds = [[3, 0], [1, 2]]
            else:
                horizontal_edge_inds = [[3, 0], [1, 2]]
                vertical_edge_inds = [[0, 1], [2, 3]]

            vertical_len_sum = norm(points[vertical_edge_inds[0][0]] -
                                    points[vertical_edge_inds[0][1]]) + norm(
                points[vertical_edge_inds[1][0]] -
                points[vertical_edge_inds[1][1]])
            horizontal_len_sum = norm(
                points[horizontal_edge_inds[0][0]] -
                points[horizontal_edge_inds[0][1]]) + norm(
                points[horizontal_edge_inds[1][0]] -
                points[horizontal_edge_inds[1][1]])

            if vertical_len_sum > horizontal_len_sum * orientation_thr:
                head_inds = horizontal_edge_inds[0]
                tail_inds = horizontal_edge_inds[1]
            else:
                head_inds = vertical_edge_inds[0]
                tail_inds = vertical_edge_inds[1]

        return head_inds, tail_inds

    def resample_sidelines(self, sideline1, sideline2, resample_step):
        """Resample two sidelines to be of the same points number according to
        step size.

        Args:
            sideline1 (ndarray): The points composing a sideline of a text
                polygon.
            sideline2 (ndarray): The points composing another sideline of a
                text polygon.
            resample_step (float): The resampled step size.

        Returns:
            resampled_line1 (ndarray): The resampled line 1.
            resampled_line2 (ndarray): The resampled line 2.
        """

        assert sideline1.ndim == sideline2.ndim == 2
        assert sideline1.shape[1] == sideline2.shape[1] == 2
        assert sideline1.shape[0] >= 2
        assert sideline2.shape[0] >= 2
        assert isinstance(resample_step, float)

        length1 = sum([
            norm(sideline1[i + 1] - sideline1[i])
            for i in range(len(sideline1) - 1)
        ])
        length2 = sum([
            norm(sideline2[i + 1] - sideline2[i])
            for i in range(len(sideline2) - 1)
        ])

        total_length = (length1 + length2) / 2
        resample_point_num = max(int(float(total_length) / resample_step), 1)

        resampled_line1 = self.resample_line(sideline1, resample_point_num)
        resampled_line2 = self.resample_line(sideline2, resample_point_num)

        return resampled_line1, resampled_line2

    def generate_center_region_mask(self, img_size, text_polys):
        """Generate text center region mask.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            center_region_mask (ndarray): The text center region mask.
        """

        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size

        center_region_mask = np.zeros((h, w), np.uint8)

        center_region_boxes = []
        for poly in text_polys:
            # assert len(poly) == 1
            polygon_points = poly.reshape(-1, 2)
            _, _, top_line, bot_line = self.reorder_poly_edge(polygon_points)
            resampled_top_line, resampled_bot_line = self.resample_sidelines(
                top_line, bot_line, self.resample_step)
            resampled_bot_line = resampled_bot_line[::-1]
            center_line = (resampled_top_line + resampled_bot_line) / 2

            line_head_shrink_len = norm(resampled_top_line[0] -
                                        resampled_bot_line[0]) / 4.0
            line_tail_shrink_len = norm(resampled_top_line[-1] -
                                        resampled_bot_line[-1]) / 4.0
            head_shrink_num = int(line_head_shrink_len // self.resample_step)
            tail_shrink_num = int(line_tail_shrink_len // self.resample_step)
            if len(center_line) > head_shrink_num + tail_shrink_num + 2:
                center_line = center_line[head_shrink_num:len(center_line) -
                                                          tail_shrink_num]
                resampled_top_line = resampled_top_line[
                                     head_shrink_num:len(resampled_top_line) - tail_shrink_num]
                resampled_bot_line = resampled_bot_line[
                                     head_shrink_num:len(resampled_bot_line) - tail_shrink_num]

            for i in range(0, len(center_line) - 1):
                tl = center_line[i] + (resampled_top_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                tr = center_line[i + 1] + (
                        resampled_top_line[i + 1] -
                        center_line[i + 1]) * self.center_region_shrink_ratio
                br = center_line[i + 1] + (
                        resampled_bot_line[i + 1] -
                        center_line[i + 1]) * self.center_region_shrink_ratio
                bl = center_line[i] + (resampled_bot_line[i] - center_line[i]
                                       ) * self.center_region_shrink_ratio
                current_center_box = np.vstack([tl, tr, br,
                                                bl]).astype(np.int32)
                center_region_boxes.append(current_center_box)

        cv2.fillPoly(center_region_mask, center_region_boxes, 1)
        return center_region_mask

    def resample_polygon(self, polygon, n=400):
        """Resample one polygon with n points on its boundary.

        Args:
            polygon (list[float]): The input polygon.
            n (int): The number of resampled points.
        Returns:
            resampled_polygon (list[float]): The resampled polygon.
        """
        length = []

        for i in range(len(polygon)):
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]
            length.append(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5)

        total_length = sum(length)
        n_on_each_line = (np.array(length) / (total_length + 1e-8)) * n
        n_on_each_line = n_on_each_line.astype(np.int32)
        new_polygon = []

        for i in range(len(polygon)):
            num = n_on_each_line[i]
            p1 = polygon[i]
            if i == len(polygon) - 1:
                p2 = polygon[0]
            else:
                p2 = polygon[i + 1]

            if num == 0:
                continue

            dxdy = (p2 - p1) / num
            for j in range(num):
                point = p1 + dxdy * j
                new_polygon.append(point)

        return np.array(new_polygon)

    def normalize_polygon(self, polygon):
        """Normalize one polygon so that its start point is at right most.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon with start point at right.
        """
        temp_polygon = polygon - polygon.mean(axis=0)
        x = np.abs(temp_polygon[:, 0])
        y = temp_polygon[:, 1]
        index_x = np.argsort(x)
        index_y = np.argmin(y[index_x[:8]])
        index = index_x[index_y]
        new_polygon = np.concatenate([polygon[index:], polygon[:index]])
        return new_polygon

    def poly2fourier(self, polygon, fourier_degree):
        """Perform Fourier transformation to generate Fourier coefficients ck
        from polygon.

        Args:
            polygon (ndarray): An input polygon.
            fourier_degree (int): The maximum Fourier degree K.
        Returns:
            c (ndarray(complex)): Fourier coefficients.
        """
        points = polygon[:, 0] + polygon[:, 1] * 1j
        c_fft = fft(points) / len(points)
        c = np.hstack((c_fft[-fourier_degree:], c_fft[:fourier_degree + 1]))
        return c

    def clockwise(self, c, fourier_degree):
        """Make sure the polygon reconstructed from Fourier coefficients c in
        the clockwise direction.

        Args:
            polygon (list[float]): The origin polygon.
        Returns:
            new_polygon (lost[float]): The polygon in clockwise point order.
        """
        if np.abs(c[fourier_degree + 1]) > np.abs(c[fourier_degree - 1]):
            return c
        elif np.abs(c[fourier_degree + 1]) < np.abs(c[fourier_degree - 1]):
            return c[::-1]
        else:
            if np.abs(c[fourier_degree + 2]) > np.abs(c[fourier_degree - 2]):
                return c
            else:
                return c[::-1]

    def cal_fourier_signature(self, polygon, fourier_degree):
        """Calculate Fourier signature from input polygon.

        Args:
              polygon (ndarray): The input polygon.
              fourier_degree (int): The maximum Fourier degree K.
        Returns:
              fourier_signature (ndarray): An array shaped (2k+1, 2) containing
                  real part and image part of 2k+1 Fourier coefficients.
        """
        resampled_polygon = self.resample_polygon(polygon)
        resampled_polygon = self.normalize_polygon(resampled_polygon)

        fourier_coeff = self.poly2fourier(resampled_polygon, fourier_degree)
        fourier_coeff = self.clockwise(fourier_coeff, fourier_degree)

        real_part = np.real(fourier_coeff).reshape((-1, 1))
        image_part = np.imag(fourier_coeff).reshape((-1, 1))
        fourier_signature = np.hstack([real_part, image_part])

        return fourier_signature

    def generate_fourier_maps(self, img_size, text_polys):
        """Generate Fourier coefficient maps.

        Args:
            img_size (tuple): The image size of (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            fourier_real_map (ndarray): The Fourier coefficient real part maps.
            fourier_image_map (ndarray): The Fourier coefficient image part
                maps.
        """

        assert isinstance(img_size, tuple)
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        k = self.fourier_degree
        real_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)
        imag_map = np.zeros((k * 2 + 1, h, w), dtype=np.float32)

        for poly in text_polys:
            # assert len(poly) == 1
            # text_instance = [[poly[i], poly[i + 1]]
            #                  for i in range(0, len(poly), 2)]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygon = np.array(poly).reshape((1, -1, 2))
            cv2.fillPoly(mask, polygon.astype(np.int32), 1)
            fourier_coeff = self.cal_fourier_signature(polygon[0], k)
            for i in range(-k, k + 1):
                if i != 0:
                    real_map[i + k, :, :] = mask * fourier_coeff[i + k, 0] + (
                            1 - mask) * real_map[i + k, :, :]
                    imag_map[i + k, :, :] = mask * fourier_coeff[i + k, 1] + (
                            1 - mask) * imag_map[i + k, :, :]
                else:
                    yx = np.argwhere(mask > 0.5)
                    k_ind = np.ones((len(yx)), dtype=np.int32) * k
                    y, x = yx[:, 0], yx[:, 1]
                    real_map[k_ind, y, x] = fourier_coeff[k, 0] - x
                    imag_map[k_ind, y, x] = fourier_coeff[k, 1] - y

        return real_map, imag_map

    def generate_text_region_mask(self, img_size, text_polys):
        """Generate text center region mask and geometry attribute maps.

        Args:
            img_size (tuple): The image size (height, width).
            text_polys (list[list[ndarray]]): The list of text polygons.

        Returns:
            text_region_mask (ndarray): The text region mask.
        """

        assert isinstance(img_size, (tuple, list))
        # assert check_argument.is_2dlist(text_polys)

        h, w = img_size
        text_region_mask = np.zeros((h, w), dtype=np.uint8)

        for poly in text_polys:
            # assert len(poly) == 1
            # text_instance = [[poly[i], poly[i + 1]]
            #                  for i in range(0, len(poly), 2)]
            polygon = np.array(
                poly, dtype=np.int32).reshape((1, -1, 2))
            cv2.fillPoly(text_region_mask, polygon, 1)

        return text_region_mask

    def generate_effective_mask(self, mask_size: tuple, polygons_ignore):
        """Generate effective mask by setting the ineffective regions to 0 and
        effective regions to 1.

        Args:
            mask_size (tuple): The mask size.
            polygons_ignore (list[[ndarray]]: The list of ignored text
                polygons.

        Returns:
            mask (ndarray): The effective mask of (height, width).
        """

        # assert check_argument.is_2dlist(polygons_ignore)

        mask = np.ones(mask_size, dtype=np.uint8)

        for poly in polygons_ignore:
            instance = poly.reshape(-1,
                                    2).astype(np.int32).reshape(1, -1, 2)
            cv2.fillPoly(mask, instance, 0)

        return mask

    def generate_level_targets(self, img_size, text_polys, ignore_polys):
        """Generate ground truth target on each level.

        Args:
            img_size (list[int]): Shape of input image.
            text_polys (list[list[ndarray]]): A list of ground truth polygons.
            ignore_polys (list[list[ndarray]]): A list of ignored polygons.
        Returns:
            level_maps (list(ndarray)): A list of ground target on each level.
        """
        h, w = img_size
        lv_size_divs = self.level_size_divisors
        lv_proportion_range = self.level_proportion_range
        lv_text_polys = [[] for i in range(len(lv_size_divs))]
        lv_ignore_polys = [[] for i in range(len(lv_size_divs))]
        level_maps = []
        for poly in text_polys:
            # assert len(poly) == 1
            # text_instance = [[poly[i], poly[i + 1]]
            #                  for i in range(0, len(poly), 2)]
            polygon = np.array(poly, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_text_polys[ind].append(poly / lv_size_divs[ind])

        for ignore_poly in ignore_polys:
            # assert len(ignore_poly) == 1
            # text_instance = [[ignore_poly[i], ignore_poly[i + 1]]
            #                  for i in range(0, len(ignore_poly), 2)]
            polygon = np.array(ignore_poly, dtype=np.int).reshape((1, -1, 2))
            _, _, box_w, box_h = cv2.boundingRect(polygon)
            proportion = max(box_h, box_w) / (h + 1e-8)

            for ind, proportion_range in enumerate(lv_proportion_range):
                if proportion_range[0] < proportion < proportion_range[1]:
                    lv_ignore_polys[ind].append(
                        ignore_poly / lv_size_divs[ind])

        for ind, size_divisor in enumerate(lv_size_divs):
            current_level_maps = []
            level_img_size = (h // size_divisor, w // size_divisor)

            text_region = self.generate_text_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(text_region)

            center_region = self.generate_center_region_mask(
                level_img_size, lv_text_polys[ind])[None]
            current_level_maps.append(center_region)

            effective_mask = self.generate_effective_mask(
                level_img_size, lv_ignore_polys[ind])[None]
            current_level_maps.append(effective_mask)

            fourier_real_map, fourier_image_maps = self.generate_fourier_maps(
                level_img_size, lv_text_polys[ind])
            current_level_maps.append(fourier_real_map)
            current_level_maps.append(fourier_image_maps)

            level_maps.append(np.concatenate(current_level_maps))

        return level_maps

    def generate_targets(self, results):
        """Generate the ground truth targets for FCENet.

        Args:
            results (dict): The input result dictionary.

        Returns:
            results (dict): The output result dictionary.
        """

        assert isinstance(results, dict)
        image = results['image']
        h, w, _ = image.shape

        if 'polygons1' not in results:
            results['polygons1'] = [results['polys'][i] for i in range(len(results['ignore_tags'])) if
                                   not results['ignore_tags'][i]]
            results['ignore_polygons1'] = [results['polys'][i] for i in range(len(results['ignore_tags'])) if
                                          results['ignore_tags'][i]]

        polygon_masks = results['polygons']
        polygon_masks_ignore = results['ignore_polygons']

        level_maps = self.generate_level_targets((h, w), polygon_masks,
                                                 polygon_masks_ignore)

        # results['mask_fields'].clear()  # rm gt_masks encoded by polygons
        # import remote_pdb as pdb;pdb.set_trace()
        mapping = {
            'p3_maps': level_maps[0],
            'p4_maps': level_maps[1],
            'p5_maps': level_maps[2]
        }
        for key, value in mapping.items():
            results[key] = value

        return results

    def __call__(self, results):
        results = self.generate_targets(results)
        return results
