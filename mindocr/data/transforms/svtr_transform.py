import copy
import logging
import math
import numbers
import random
from typing import Callable, Iterable

import cv2
import numpy as np

from mindspore.dataset.vision import RandomColorAdjust

__all__ = ["SVTRRecAug", "MultiLabelEncode", "RecConAug", "RecAug", "RecResizeImgForSVTR"]

_logger = logging.getLogger(__name__)


class Compose:
    def __init__(self, funcs: Iterable[Callable]) -> None:
        self.funcs = funcs

    def __call__(self, x):
        for fn in self.funcs:
            x = fn(x)
        return x


def sample_asym(magnitude, size=None):
    return np.random.beta(1, 4, size) * magnitude


def sample_sym(magnitude, size=None):
    return (np.random.beta(4, 4, size=size) - 0.5) * 2 * magnitude


def sample_uniform(low, high, size=None):
    return np.random.uniform(low, high, size=size)


def get_interpolation(type="random"):
    if type == "random":
        choice = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]
        interpolation = choice[random.randint(0, len(choice) - 1)]
    elif type == "nearest":
        interpolation = cv2.INTER_NEAREST
    elif type == "linear":
        interpolation = cv2.INTER_LINEAR
    elif type == "cubic":
        interpolation = cv2.INTER_CUBIC
    elif type == "area":
        interpolation = cv2.INTER_AREA
    else:
        raise TypeError("Interpolation types only nearest, linear, cubic, area are supported!")
    return interpolation


class CVRandomRotation(object):
    def __init__(self, degrees=15):
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        return sample_sym(degrees)

    def __call__(self, img):
        angle = self.get_params(self.degrees)
        src_h, src_w = img.shape[:2]
        M = cv2.getRotationMatrix2D(center=(src_w / 2, src_h / 2), angle=angle, scale=1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        dst_w = int(src_h * abs_sin + src_w * abs_cos)
        dst_h = int(src_h * abs_cos + src_w * abs_sin)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class CVRandomAffine(object):
    def __init__(self, degrees, translate=None, scale=None, shear=None):
        assert isinstance(degrees, numbers.Number), "degree should be a single number."
        assert degrees >= 0, "degree must be positive."
        self.degrees = degrees

        if translate is not None:
            assert (
                isinstance(translate, (tuple, list)) and len(translate) == 2
            ), "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert (
                isinstance(scale, (tuple, list)) and len(scale) == 2
            ), "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = [shear]
            else:
                assert isinstance(shear, (tuple, list)) and (
                    len(shear) == 2
                ), "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        if not isinstance(shear, (tuple, list)) and len(shear) == 2:
            raise ValueError(
                "Shear should be a single value or a tuple/list containing " + "two values. Got {}".format(shear)
            )

        rot = math.radians(angle)
        sx, sy = [math.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = np.cos(rot - sy) / np.cos(sy)
        b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
        c = np.sin(rot - sy) / np.cos(sy)
        d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, height):
        angle = sample_sym(degrees)
        if translate is not None:
            max_dx = translate[0] * height
            max_dy = translate[1] * height
            translations = (np.round(sample_sym(max_dx)), np.round(sample_sym(max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = sample_uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            if len(shears) == 1:
                shear = [sample_sym(shears[0]), 0.0]
            elif len(shears) == 2:
                shear = [sample_sym(shears[0]), sample_sym(shears[1])]
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        src_h, src_w = img.shape[:2]
        angle, translate, scale, shear = self.get_params(self.degrees, self.translate, self.scale, self.shear, src_h)

        M = self._get_inverse_affine_matrix((src_w / 2, src_h / 2), angle, (0, 0), scale, shear)
        M = np.array(M).reshape(2, 3)

        startpoints = [(0, 0), (src_w - 1, 0), (src_w - 1, src_h - 1), (0, src_h - 1)]

        def project(x, y, a, b, c):
            return int(a * x + b * y + c)

        endpoints = [(project(x, y, *M[0]), project(x, y, *M[1])) for x, y in startpoints]

        rect = cv2.minAreaRect(np.array(endpoints))
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()

        dst_w = int(max_x - min_x)
        dst_h = int(max_y - min_y)
        M[0, 2] += (dst_w - src_w) / 2
        M[1, 2] += (dst_h - src_h) / 2

        # add translate
        dst_w += int(abs(translate[0]))
        dst_h += int(abs(translate[1]))
        if translate[0] < 0:
            M[0, 2] += abs(translate[0])
        if translate[1] < 0:
            M[1, 2] += abs(translate[1])

        flags = get_interpolation()
        return cv2.warpAffine(img, M, (dst_w, dst_h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


class CVRandomPerspective(object):
    def __init__(self, distortion=0.5):
        self.distortion = distortion

    def get_params(self, width, height, distortion):
        offset_h = sample_asym(distortion * height / 2, size=4).astype(dtype=np.int32)
        offset_w = sample_asym(distortion * width / 2, size=4).astype(dtype=np.int32)
        topleft = (offset_w[0], offset_h[0])
        topright = (width - 1 - offset_w[1], offset_h[1])
        botright = (width - 1 - offset_w[2], height - 1 - offset_h[2])
        botleft = (offset_w[3], height - 1 - offset_h[3])

        startpoints = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return np.array(startpoints, dtype=np.float32), np.array(endpoints, dtype=np.float32)

    def __call__(self, img):
        height, width = img.shape[:2]
        startpoints, endpoints = self.get_params(width, height, self.distortion)
        M = cv2.getPerspectiveTransform(startpoints, endpoints)

        # TODO: more robust way to crop image
        rect = cv2.minAreaRect(endpoints)
        bbox = cv2.boxPoints(rect).astype(dtype=np.int32)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
        min_x, min_y = max(min_x, 0), max(min_y, 0)

        flags = get_interpolation()
        img = cv2.warpPerspective(img, M, (max_x, max_y), flags=flags, borderMode=cv2.BORDER_REPLICATE)
        img = img[min_y:, min_x:]
        return img


class CVRescale(object):
    def __init__(self, factor=4, base_size=(128, 512)):
        """Define image scales using gaussian pyramid and rescale image to target scale.

        Args:
            factor: the decayed factor from base size, factor=4 keeps target scale by default.
            base_size: base size the build the bottom layer of pyramid
        """
        # assert factor is valid
        self.factor = factor
        self.base_h, self.base_w = base_size[:2]

    def __call__(self, img):
        if isinstance(self.factor, numbers.Number):
            factor = round(sample_uniform(0, self.factor))
        elif isinstance(self.factor, (tuple, list)) and len(self.factor) == 2:
            factor = round(sample_uniform(self.factor[0], self.factor[1]))
        else:
            raise RuntimeError("factor must be number or list with length 2")

        if factor == 0:
            return img
        src_h, src_w = img.shape[:2]
        cur_w, cur_h = self.base_w, self.base_h
        scale_img = cv2.resize(img, (cur_w, cur_h), interpolation=get_interpolation())
        for _ in range(factor):
            scale_img = cv2.pyrDown(scale_img)
        scale_img = cv2.resize(scale_img, (src_w, src_h), interpolation=get_interpolation())
        return scale_img


class CVGaussianNoise(object):
    def __init__(self, mean=0, variance=20):
        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        if isinstance(self.variance, numbers.Number):
            variance = max(int(sample_asym(self.variance)), 1)
        elif isinstance(self.variance, (tuple, list)) and len(self.variance) == 2:
            variance = int(sample_uniform(self.variance[0], self.variance[1]))
        else:
            raise RuntimeError("degree must be number or list with length 2")

        noise = np.random.normal(self.mean, variance**0.5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img


class CVMotionBlur(object):
    def __init__(self, degrees=12, angle=90):
        self.degrees = degrees
        self.angle = angle

    def __call__(self, img):
        if isinstance(self.degrees, numbers.Number):
            degree = max(int(sample_asym(self.degrees)), 1)
        elif isinstance(self.degrees, (tuple, list)) and len(self.degrees) == 2:
            degree = int(sample_uniform(self.degrees[0], self.degrees[1]))
        else:
            raise RuntimeError("degree must be number or list with length 2")
        angle = sample_uniform(-self.angle, self.angle)

        M = cv2.getRotationMatrix2D((degree // 2, degree // 2), angle, 1)
        motion_blur_kernel = np.zeros((degree, degree))
        motion_blur_kernel[degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        img = cv2.filter2D(img, -1, motion_blur_kernel)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img


class CVColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.5):
        self.p = p
        self.transforms = RandomColorAdjust(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transforms(img)
        else:
            return img


class SVTRDeterioration(object):
    def __init__(self, variance, degrees, factor, p=0.5):
        self.p = p
        transforms = []
        if variance is not None:
            transforms.append(CVGaussianNoise(variance=variance))
        if degrees is not None:
            transforms.append(CVMotionBlur(degrees=degrees))
        if factor is not None:
            transforms.append(CVRescale(factor=factor))
        self.transforms = transforms

    def __call__(self, img):
        if random.random() < self.p:
            random.shuffle(self.transforms)
            transforms = Compose(self.transforms)
            return transforms(img)
        else:
            return img


class SVTRGeometry(object):
    def __init__(
        self,
        aug_type=0,
        degrees=15,
        translate=(0.3, 0.3),
        scale=(0.5, 2.0),
        shear=(45, 15),
        distortion=0.5,
        p=0.5,
    ):
        self.aug_type = aug_type
        self.p = p
        self.transforms = []
        self.transforms.append(CVRandomRotation(degrees=degrees))
        self.transforms.append(CVRandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear))
        self.transforms.append(CVRandomPerspective(distortion=distortion))

    def __call__(self, img):
        if random.random() < self.p:
            if self.aug_type:
                random.shuffle(self.transforms)
                transforms = Compose(self.transforms[: random.randint(1, 3)])
                img = transforms(img)
            else:
                img = self.transforms[random.randint(0, 2)](img)
            return img
        else:
            return img


class SVTRRecAug(object):
    def __init__(
        self, aug_type=0, geometry_p=0.5, deterioration_p=0.25, deterioration_factor=4, colorjitter_p=0.25, **kwargs
    ):
        self.transforms = Compose(
            [
                SVTRGeometry(
                    aug_type=aug_type,
                    degrees=45,
                    translate=(0.0, 0.0),
                    scale=(0.5, 2.0),
                    shear=(45, 15),
                    distortion=0.5,
                    p=geometry_p,
                ),
                SVTRDeterioration(variance=20, degrees=6, factor=deterioration_factor, p=deterioration_p),
                CVColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1,
                    p=colorjitter_p,
                ),
            ]
        )

    def __call__(self, data):
        img = data["image"]
        img = self.transforms(img)
        data["image"] = img
        return data


class BaseRecLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, lower=False):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower

        if character_dict_path is None:
            _logger.warning("The character_dict_path is None, model can only recognize number and lower letters")
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
        else:
            self.character_str = []
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            if use_space_char:
                self.character_str.append(" ")
            dict_character = list(self.character_str)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.lower:
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                _logger.warning("{} is not in dict".format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncodeForSVTR(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs):
        super(CTCLabelEncodeForSVTR, self).__init__(max_text_length, character_dict_path, use_space_char)

    def __call__(self, data):
        text = data["label"]
        text_str = text
        text = self.encode(text)
        if text is None:
            return None
        data["text_length"] = np.array(len(text_str))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)

        label = [0] * len(self.character)
        for x in text:
            label[x] += 1
        data["label_ace"] = np.array(label)
        data["text_padded"] = text_str + " " * (self.max_text_len - len(text_str))
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class SARLabelEncodeForSVTR(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs):
        super(SARLabelEncodeForSVTR, self).__init__(max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        beg_end_str = "<BOS/EOS>"
        unknown_str = "<UKN>"
        padding_str = "<PAD>"
        dict_character = dict_character + [unknown_str]
        self.unknown_idx = len(dict_character) - 1
        dict_character = dict_character + [beg_end_str]
        self.start_idx = len(dict_character) - 1
        self.end_idx = len(dict_character) - 1
        dict_character = dict_character + [padding_str]
        self.padding_idx = len(dict_character) - 1

        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["length"] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[: len(target)] = target
        data["label"] = np.array(padded_text)
        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


class MultiLabelEncode(BaseRecLabelEncode):
    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, gtc_encode=None, **kwargs):
        super(MultiLabelEncode, self).__init__(max_text_length, character_dict_path, use_space_char)

        self.ctc_encode = CTCLabelEncodeForSVTR(max_text_length, character_dict_path, use_space_char, **kwargs)
        self.gtc_encode_type = gtc_encode
        if gtc_encode is None:
            self.gtc_encode = SARLabelEncodeForSVTR(max_text_length, character_dict_path, use_space_char, **kwargs)
        else:
            self.gtc_encode = eval(gtc_encode)(max_text_length, character_dict_path, use_space_char, **kwargs)

    def __call__(self, data):
        data_ctc = copy.deepcopy(data)
        data_gtc = copy.deepcopy(data)
        data_out = dict()
        data_out["img_path"] = data.get("img_path", None)
        data_out["image"] = data["image"]
        ctc = self.ctc_encode.__call__(data_ctc)
        gtc = self.gtc_encode.__call__(data_gtc)
        if ctc is None or gtc is None:
            return None
        data_out["label_ctc"] = ctc["label"].astype("int32")
        if self.gtc_encode_type is not None:
            data_out["label_gtc"] = gtc["label"].astype("int32")
        else:
            data_out["label_sar"] = gtc["label"].astype("int32")
        data_out["text_length"] = ctc["text_length"].astype("int32")
        data_out["text_padded"] = ctc["text_padded"]
        return data_out


class RecConAug(object):
    def __init__(self, prob=0.5, image_shape=(32, 320, 3), max_text_length=25, ext_data_num=1, **kwargs):
        self.ext_data_num = ext_data_num
        self.prob = prob
        self.max_text_length = max_text_length
        self.image_shape = image_shape
        self.max_wh_ratio = self.image_shape[1] / self.image_shape[0]

    def merge_ext_data(self, data, ext_data):
        ori_w = round(data["image"].shape[1] / data["image"].shape[0] * self.image_shape[0])
        ext_w = round(ext_data["image"].shape[1] / ext_data["image"].shape[0] * self.image_shape[0])
        data["image"] = cv2.resize(data["image"], (ori_w, self.image_shape[0]))
        ext_data["image"] = cv2.resize(ext_data["image"], (ext_w, self.image_shape[0]))
        data["image"] = np.concatenate([data["image"], ext_data["image"]], axis=1)
        data["label"] += ext_data["label"]
        return data

    def __call__(self, data):
        rnd_num = random.random()
        if rnd_num > self.prob:
            return data
        for idx, ext_data in enumerate(data["ext_data"]):
            if len(data["label"]) + len(ext_data["label"]) > self.max_text_length:
                break
            concat_ratio = (
                data["image"].shape[1] / data["image"].shape[0]
                + ext_data["image"].shape[1] / ext_data["image"].shape[0]
            )
            if concat_ratio > self.max_wh_ratio:
                break
            data = self.merge_ext_data(data, ext_data)
        data.pop("ext_data")
        return data


def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0 : h - top_crop, :, :]
    return crop_img


def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


def hsv_aug(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[: w - i, : h - i, :]
        return img
    else:
        return img


def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


class WarpMLS:
    def __init__(self, src, src_pts, dst_pts, dst_w, dst_h, trans_ratio=1.0):
        self.src = src
        self.src_pts = src_pts
        self.dst_pts = dst_pts
        self.pt_count = len(self.dst_pts)
        self.dst_w = dst_w
        self.dst_h = dst_h
        self.trans_ratio = trans_ratio
        self.grid_size = 100
        self.rdx = np.zeros((self.dst_h, self.dst_w))
        self.rdy = np.zeros((self.dst_h, self.dst_w))

    @staticmethod
    def __bilinear_interp(x, y, v11, v12, v21, v22):
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 * (1 - y) + v22 * y) * x

    def generate(self):
        self.calc_delta()
        return self.gen_img()

    def calc_delta(self):
        w = np.zeros(self.pt_count, dtype=np.float32)

        if self.pt_count < 2:
            return

        i = 0
        while 1:
            if self.dst_w <= i < self.dst_w + self.grid_size - 1:
                i = self.dst_w - 1
            elif i >= self.dst_w:
                break

            j = 0
            while 1:
                if self.dst_h <= j < self.dst_h + self.grid_size - 1:
                    j = self.dst_h - 1
                elif j >= self.dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(self.pt_count):
                    if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                        break

                    w[k] = 1.0 / (
                        (i - self.dst_pts[k][0]) * (i - self.dst_pts[k][0])
                        + (j - self.dst_pts[k][1]) * (j - self.dst_pts[k][1])
                    )

                    sw += w[k]
                    swp = swp + w[k] * np.array(self.dst_pts[k])
                    swq = swq + w[k] * np.array(self.src_pts[k])

                if k == self.pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue
                        pt_i = self.dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(self.pt_count):
                        if i == self.dst_pts[k][0] and j == self.dst_pts[k][1]:
                            continue

                        pt_i = self.dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = (
                            np.sum(pt_i * cur_pt) * self.src_pts[k][0] - np.sum(pt_j * cur_pt) * self.src_pts[k][1]
                        )
                        tmp_pt[1] = (
                            -np.sum(pt_i * cur_pt_j) * self.src_pts[k][0] + np.sum(pt_j * cur_pt_j) * self.src_pts[k][1]
                        )
                        tmp_pt *= w[k] / miu_s
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = self.src_pts[k]

                self.rdx[j, i] = new_pt[0] - i
                self.rdy[j, i] = new_pt[1] - j

                j += self.grid_size
            i += self.grid_size

    def gen_img(self):
        src_h, src_w = self.src.shape[:2]
        dst = np.zeros_like(self.src, dtype=np.float32)

        for i in np.arange(0, self.dst_h, self.grid_size):
            for j in np.arange(0, self.dst_w, self.grid_size):
                ni = i + self.grid_size
                nj = j + self.grid_size
                w = h = self.grid_size
                if ni >= self.dst_h:
                    ni = self.dst_h - 1
                    h = ni - i + 1
                if nj >= self.dst_w:
                    nj = self.dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self.__bilinear_interp(
                    di / h, dj / w, self.rdx[i, j], self.rdx[i, nj], self.rdx[ni, j], self.rdx[ni, nj]
                )
                delta_y = self.__bilinear_interp(
                    di / h, dj / w, self.rdy[i, j], self.rdy[i, nj], self.rdy[ni, j], self.rdy[ni, nj]
                )
                nx = j + dj + delta_x * self.trans_ratio
                ny = i + di + delta_y * self.trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(self.src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i : i + h, j : j + w] = self.__bilinear_interp(
                    x, y, self.src[nyi, nxi], self.src[nyi, nxi1], self.src[nyi1, nxi], self.src[nyi1, nxi1]
                )

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst


def tia_distort(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append(
            [cut * cut_idx + np.random.randint(thresh) - half_thresh, np.random.randint(thresh) - half_thresh]
        )
        dst_pts.append(
            [cut * cut_idx + np.random.randint(thresh) - half_thresh, img_h + np.random.randint(thresh) - half_thresh]
        )

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_stretch(src, segment=4):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def tia_perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


class BaseDataAugmentation(object):
    def __init__(
        self,
        crop_prob=0.4,
        reverse_prob=0.4,
        noise_prob=0.4,
        jitter_prob=0.4,
        blur_prob=0.4,
        hsv_aug_prob=0.4,
        **kwargs
    ):
        self.crop_prob = crop_prob
        self.reverse_prob = reverse_prob
        self.noise_prob = noise_prob
        self.jitter_prob = jitter_prob
        self.blur_prob = blur_prob
        self.hsv_aug_prob = hsv_aug_prob
        # for GaussianBlur
        self.fil = cv2.getGaussianKernel(ksize=5, sigma=1, ktype=cv2.CV_32F)

    def __call__(self, data):
        img = data["image"]
        h, w, _ = img.shape

        if random.random() <= self.crop_prob and h >= 20 and w >= 20:
            img = get_crop(img)

        if random.random() <= self.blur_prob:
            # GaussianBlur
            img = cv2.sepFilter2D(img, -1, self.fil, self.fil)

        if random.random() <= self.hsv_aug_prob:
            img = hsv_aug(img)

        if random.random() <= self.jitter_prob:
            img = jitter(img)

        if random.random() <= self.noise_prob:
            img = add_gasuss_noise(img)

        if random.random() <= self.reverse_prob:
            img = 255 - img

        data["image"] = img
        return data


class RecAug(object):
    def __init__(
        self,
        tia_prob=0.4,
        crop_prob=0.4,
        reverse_prob=0.4,
        noise_prob=0.4,
        jitter_prob=0.4,
        blur_prob=0.4,
        hsv_aug_prob=0.4,
        **kwargs
    ):
        self.tia_prob = tia_prob
        self.bda = BaseDataAugmentation(crop_prob, reverse_prob, noise_prob, jitter_prob, blur_prob, hsv_aug_prob)

    def __call__(self, data):
        img = data["image"]
        h, w, _ = img.shape

        # tia
        if random.random() <= self.tia_prob:
            if h >= 20 and w >= 20:
                img = tia_distort(img, random.randint(3, 6))
                img = tia_stretch(img, random.randint(3, 6))
            img = tia_perspective(img)

        # bda
        data["image"] = img
        data = self.bda(data)
        return data


def resize_norm_img_chinese(img, image_shape, width_downsample_ratio):
    imgC, imgH, imgW = image_shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))

    # generate valid_width_mask
    width_downsampled = int(image_shape[-1] * width_downsample_ratio)
    valid_width_mask = np.full([1, width_downsampled], 1).astype("int32")
    valid_width = min(width_downsampled, int(width_downsampled * valid_ratio))
    valid_width_mask[:, valid_width:] = 0

    return padding_im, valid_ratio, valid_width_mask


def resize_norm_img(
    img,
    image_shape,
    padding=True,
    interpolation=cv2.INTER_LINEAR,
    width_downsample_ratio=0.125,
):
    imgC, imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    if not padding:
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype("float32")
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))

    # generate valid_width_mask
    width_downsampled = int(image_shape[-1] * width_downsample_ratio)
    valid_width_mask = np.full([1, width_downsampled], 1).astype("int32")
    valid_width = min(width_downsampled, int(width_downsampled * valid_ratio))
    valid_width_mask[:, valid_width:] = 0

    return padding_im, valid_ratio, valid_width_mask


class RecResizeImgForSVTR(object):
    def __init__(
        self,
        image_shape,
        infer_mode=False,
        eval_mode=False,
        character_dict_path=".mindocr/utils/dict/ch_dict.txt",
        padding=True,
        width_downsample_ratio=0.125,
        **kwargs
    ):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.eval_mode = eval_mode
        self.character_dict_path = character_dict_path
        self.padding = padding
        self.width_downsample_ratio = width_downsample_ratio

    def __call__(self, data):
        img = data["image"]
        if self.eval_mode or (self.infer_mode and self.character_dict_path is not None):
            norm_img, valid_ratio, valid_width_mask = resize_norm_img_chinese(
                img=img, image_shape=self.image_shape, width_downsample_ratio=self.width_downsample_ratio
            )
        else:
            norm_img, valid_ratio, valid_width_mask = resize_norm_img(
                img=img,
                image_shape=self.image_shape,
                padding=self.padding,
                width_downsample_ratio=self.width_downsample_ratio,
            )
        data["image"] = norm_img
        data["valid_ratio"] = valid_ratio
        data["valid_width_mask"] = valid_width_mask
        return data
