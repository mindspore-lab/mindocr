"""
transforms adopted from model zoo.
"""
import cv2
import math
import numpy as np
from mindspore.dataset.vision import RandomColorAdjust, ToPIL
from shapely.geometry import Polygon
import pyclipper


__all__ = ['MZMakeSegDetectionData', 'MZRandomColorAdjust', 'MZScalePad', 'MZResizeByGrid', 'MZRandomCropData', 'MZRandomScaleByShortSide']



class MZMakeSegDetectionData:
    """
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    """
    def __init__(self, min_text_size=8, shrink_ratio=0.4, is_training=True):
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.is_training = is_training

    #def process(self, img, polys, dontcare):

    def __call__(self, data):
        """
        required keys:
            image, polys, ignore_tags
        added keys:
            shrink_map: text region bit map
            shrink_mask: ignore mask, pexels where value is 1 indicates no contribution to loss
        """
        img = data['image']
        polys = data['polys']
        dontcare = data['ignore_tags']

        h, w = img.shape[:2]
        if self.is_training:
            polys, dontcare = self.validate_polygons(polys, dontcare, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        for i in range(len(polys)):
            polygon = polys[i]
            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])
            if dontcare[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                dontcare[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                           (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polys[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                shrunk = padding.Execute(-distance)
                if shrunk == []:
                    cv2.fillPoly(mask, polygon.astype(np.int32)[np.newaxis, :, :], 0)
                    dontcare[i] = True
                    continue
                shrunk = np.array(shrunk[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrunk.astype(np.int32)], 1)

        data['binary_map'] = gt
        data['mask'] = mask
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        """polygons (numpy.array, required): of shape (num_instances, num_points, 2)"""
        if polygons is None:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)

        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] - polygon[i, 1])

        return edge / 2.



class MZRandomColorAdjust:
    def __init__(self, brightness=32.0 / 255, saturation=0.5, to_numpy=False):
        self.colorjitter = RandomColorAdjust(brightness=brightness, saturation=saturation)
        self.to_numpy = to_numpy

    def __call__(self, data):
        """
        required keys: image, numpy RGB format. (TODO: check BGR format support)
        modified keys: image, PIL / numpy
        """
        img = data['image']
        img = self.colorjitter(ToPIL()(img.astype(np.uint8)))  # return PIL
        if self.to_numpy:
            img = np.array(img)
        data['image'] = img

        return data




class MZScalePad:
    """
    scale image and polys with short side, then pad to eval_size.
    input image format: hwc
    """
    def __init__(self, eval_size=[736, 1280]):
        self.eval_size = eval_size

    def __call__(self, data):
        """
        required keys:
            image, HWC
            (polys)
        modified keys:
            image
            (polys)
        added keys:
            shape: [src_h, src_w, scale_ratio_h, scale_ratio_w]
        """
        img = data['image']
        if 'polys' in data:
            polys = data['polys']
        else:
            polys = None
        eval_size = self.eval_size

        h, w, c = img.shape
        s_h = eval_size[0] / h
        s_w = eval_size[1] / w
        scale = min(s_h, s_w)
        new_h = int(scale * h)
        new_w = int(scale * w)
        img = cv2.resize(img, (new_w, new_h))
        padimg = np.zeros((eval_size[0], eval_size[1], c), img.dtype)
        padimg[:new_h, :new_w, :] = img

        data['image'] = padimg
        # print('polys: ', type(polys))
        if polys is not None:
            polys = polys * scale
            data['polys'] = polys

        data['shape'] = np.array([h, w, scale, scale], dtype='float32')
        return data


# TODO: This can be problematic. In ModelZoo original = resize(img), (720, 1280) resized to (736, 1280), but polys are not parsed and transformed. Fixing dataset bugs?
class MZResizeByGrid:
    """
    resize image by ratio so that it's shape is align to grid of divisor
    required key in data: img in shape of (h, w, c)
    """
    def __init__(self, divisor=32, transform_polys=True):
        self.divisor = divisor
        # self.is_train = is_train
        self.transform_polys = transform_polys

    def __call__(self, data):
        img = data['image']
        if 'polys' in data and self.transform_polys:
            polys = data['polys']
        else:
            polys = None

        divisor = self.divisor
        w_scale = math.ceil(img.shape[1] / divisor) * divisor / img.shape[1]
        h_scale = math.ceil(img.shape[0] / divisor) * divisor / img.shape[0]
        img = cv2.resize(img, dsize=None, fx=w_scale, fy=h_scale)
        data['image'] = img

        if polys is None:
            return data

        # if self.is_train:
        polys[:, :, 0] = polys[:, :, 0] * w_scale
        polys[:, :, 1] = polys[:, :, 1] * h_scale

        data['polys'] = polys
        return data


def solve_polys(polys):
    """Group poly together."""
    max_points = 0
    for poly in polys:
        if len(poly) // 2 > max_points:
            max_points = len(poly) // 2
    new_polys = []
    for poly in polys:
        new_poly = []
        if len(poly) // 2 < max_points:
            new_poly.extend(poly)
            for _ in range(len(poly) // 2, max_points):
                new_poly.extend([poly[0], poly[1]])
        else:
            new_poly = poly
        new_polys.append(new_poly)
    return np.array(new_polys), max_points


class MZRandomCropData:
    """Random crop class, include many crop relevant functions."""
    def __init__(self, max_tries=10, min_crop_side_ratio=0.1, crop_size=(640, 640)):
        self.size = crop_size
        self.min_crop_side_ratio = min_crop_side_ratio
        self.max_tries = max_tries

    def __call__(self, data):
        img = data['image']
        polys = data['polys']
        dontcare = data['ignore_tags']

        # Eliminate dontcare polys.
        all_care_polys = [polys[i] for i in range(len(dontcare)) if not dontcare[i]]
        # Crop a rectangle randomly.
        crop_x, crop_y, crop_w, crop_h = self.crop_area(img, all_care_polys)
        # Rescale the cropped rectangle to crop_size.
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        # Pad the rest of crop_size with 0.
        padimg = np.zeros((self.size[1], self.size[0], img.shape[2]), img.dtype)
        padimg[:h, :w] = cv2.resize(img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
        img = padimg

        # TODO: use numpy compute, not list
        new_polys = []
        new_dontcare = []
        for i in range(len(polys)):
            # Rescale all original polys.
            poly = polys[i]
            poly = ((np.array(poly) - (crop_x, crop_y)) * scale)
            # Filter out the polys in the cropped rectangle.
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                new_polys.append(poly)
                new_dontcare.append(dontcare[i])
        new_polys = np.array(new_polys, dtype='float32')

        data['image'] = img
        data['polys'] = new_polys
        data['ignore_tags'] = new_dontcare
        return data

    def is_poly_in_rect(self, poly, x, y, w, h):
        '''
        Whether the poly is inside a rectangle.
        '''
        poly = np.array(poly)
        if poly[:, 0].min() < x or poly[:, 0].max() > x + w:
            return False
        if poly[:, 1].min() < y or poly[:, 1].max() > y + h:
            return False
        return True

    def is_poly_outside_rect(self, poly, x, y, w, h):
        '''
        Whether the poly isn't inside a rectangle.
        '''
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        '''
        Splitting out the continuous area in the axis.
        '''
        regions = []
        min_axis = 0
        for i in range(1, len(axis)):
            # If continuous
            if axis[i] != axis[i - 1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        '''
        Randomly select two values in a single region.
        '''
        xx = np.random.choice(axis, size=2)
        xmin = np.clip(np.min(xx), 0, max_size - 1)
        xmax = np.clip(np.max(xx), 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        '''
        Two regions are randomly selected from regions and then one value is taken from each.
        Return the two values taken.
        '''
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)

        xmin = np.clip(min(selected_values), 0, max_size - 1)
        xmax = np.clip(max(selected_values), 0, max_size - 1)
        return xmin, xmax

    def crop_area(self, img, polys):
        '''
        Randomly select a rectangle containing polys from the img.
        Return the start point and side lengths of the selected rectangle.
        '''
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)

        for points in polys:
            # Convert points from float to int.
            points = np.round(points, decimals=0).astype(np.int32)
            # interval of x
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            # interval of y
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # Get the idx that include text.
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if (not h_axis.any()) or (not w_axis.any()):
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for _ in range(self.max_tries):
            # Randomly select two contained idx in the axis to form a new rectangle.
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)
            # If too small, reselect.
            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                continue
            # If there is a poly inside the rectangle, successful.
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        # If the num of attempts exceeds 'max_tries', return the whole img.
        return 0, 0, w, h


class MZRandomScaleByShortSide:
    def __init__(self, short_side):
        self.short_side = short_side

    def __call__(self, data):
        """
        required keys:
            - polys: numpy array, [num_bbox, num_points, 2]
        """
        polys = data['polys']
        img = data['image']
        short_side = self.short_side

        polys, max_points = solve_polys(polys)
        h, w = img.shape[0:2]

        # print(h, w, polys, max_points)

        # polys -> polys' scale w.r.t original.
        # TODO: use np compute, not list
        polys_scale = []
        for poly in polys:
            poly = np.asarray(poly)
            # poly = poly / ([w * 1.0, h * 1.0] * max_points)
            poly = poly / [w * 1.0, h * 1.0]
            polys_scale.append(poly)
        polys_scale = np.array(polys_scale)

        # Resize to 1280 pixs max-length.
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        h, w = img.shape[0:2]

        # Get scale randomly.
        random_scale = np.array([0.5, 1.0, 2.0, 3.0])
        scale = np.random.choice(random_scale)
        # If less than short_side, scale will be clipped to min_scale.
        if min(h, w) * scale <= short_side:
            scale = (short_side + 10) * 1.0 / min(h, w)
        # Rescale img.
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
        # Rescale polys: (N, 8) -> (N, 4, 2)
        # new_polys = (polys_scale * ([img.shape[1], img.shape[0]] * max_points)).reshape((polys.shape[0], polys.shape[1] // 2, 2))

        new_polys = polys_scale * ([img.shape[1], img.shape[0]])
        # print(new_polys.shape, )

        data['image'] = img
        data['polys'] = new_polys
        return data


