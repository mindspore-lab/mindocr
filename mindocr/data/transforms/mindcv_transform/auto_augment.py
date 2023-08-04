"""
AutoAugment and RandAugment for mindspore.

Adapted from:
    https://github.com/rwightman/pytorch-image-models/blob/main/timm/data/auto_augment.py

Papers:
    AutoAugment: Learning Augmentation Policies from Data - https://arxiv.org/abs/1805.09501
    Learning Data Augmentation Strategies for Object Detection - https://arxiv.org/abs/1906.11172
    RandAugment: Practical automated data augmentation... - https://arxiv.org/abs/1909.13719
"""

import random
import re
from functools import partial

import numpy as np

from mindspore.dataset import vision
from mindspore.dataset.vision import Inter

_FILL = (128, 128, 128)

_LEVEL_DENOM = 10.0

_HPARAMS_DEFAULT = dict(
    img_mean=_FILL,
)

_RANDOM_INTERPOLATION = (Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA)
_DEFAULT_INTERPOLATION = Inter.BICUBIC

_GUASSS_KERNEL_SIZE = 3


def _interpolation(kwargs):
    interpolation = kwargs.pop("resample", _DEFAULT_INTERPOLATION)
    if isinstance(interpolation, (list, tuple)):
        return random.choice(interpolation)
    else:
        return interpolation


def _check_args_tf(kwargs):
    kwargs["resample"] = _interpolation(kwargs)


def shear_x(img, shear, **kwargs):
    _check_args_tf(kwargs)
    return vision.RandomAffine(degrees=0, shear=(-shear, -shear), **kwargs)(img)


def shear_y(img, shear, **kwargs):
    _check_args_tf(kwargs)
    return vision.RandomAffine(degrees=0, shear=(0, 0, shear, shear), **kwargs)(img)


def translate_x(img, translate, **kwargs):
    _check_args_tf(kwargs)
    return vision.RandomAffine(degrees=0, translate=(translate, translate), **kwargs)(img)


def translate_y(img, translate, **kwargs):
    _check_args_tf(kwargs)
    return vision.RandomAffine(degrees=0, translate=(0, 0, translate, translate), **kwargs)(img)


def rotate(img, degrees, **kwargs):
    _check_args_tf(kwargs)
    return vision.RandomRotation(degrees=(degrees, degrees), **kwargs)(img)


def auto_contrast(img, **__):
    return vision.AutoContrast()(img)


def invert(img, **__):
    return vision.Invert()(img)


def equalize(img, **__):
    return vision.Equalize()(img)


def solarize(img, thresh, **__):
    return vision.RandomSolarize(threshold=(thresh, thresh))(img)


def posterize(img, bits_to_keep, **__):
    bits = max(8 - bits_to_keep, 1)
    return vision.RandomPosterize(bits=(bits, bits))(img)


def contrast(img, factor, **__):
    return vision.RandomColorAdjust(contrast=(factor, factor))(img)


def color(img, degrees, **__):
    return vision.RandomColor(degrees=(degrees, degrees))(img)


def brightness(img, factor, **__):
    return vision.RandomColorAdjust(brightness=(factor, factor))(img)


def sharpness(img, degrees, **__):
    return vision.RandomSharpness(degrees=(degrees, degrees))(img)


def gaussian_blur_rand(img, factor, **__):
    radius_min = 0.1
    radius_max = 2.0
    return vision.GaussianBlur(kernel_size=_GUASSS_KERNEL_SIZE, sigma=random.uniform(radius_min, radius_max * factor))(
        img
    )


def desaturate(img, degrees, **_):
    degrees = min(1.0, max(0.0, 1.0 - degrees))
    return vision.RandomColor(degrees=(degrees, degrees))(img)


def _randomly_negate(v):
    """With 50% probability, negate the value"""
    return -v if random.random() > 0.5 else v


# fmt: off
def _rotate_level_to_arg(level, _hparams):
    # ra: range [-30, 30]; ta: range[-135, 135]
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        level_denom = _hparams.get("magnitude_max", 31)
        level = (level / level_denom) * 135.0
    else:
        level = (level / _LEVEL_DENOM) * 30.0
    level = _randomly_negate(level)
    return level,


def _enhance_level_to_arg(level, _hparams):
    # ra: range [0.1, 1.9]; ta: range[0.01, 2.00]
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        level_denom = _hparams.get("magnitude_max", 31)
        return (level / level_denom) * 1.99 + 0.01,
    return (level / _LEVEL_DENOM) * 1.8 + 0.1,


def _enhance_increasing_level_to_arg(level, _hparams):
    # the 'no change' level is 1.0, moving from 1.0 to 0. or 2.0 to increases enhanced blending
    # range [0.1, 1.9] if level <= _LEVEL_DENOM
    level = (level / _LEVEL_DENOM) * 0.9
    level = max(0.1, 1.0 + _randomly_negate(level))  # keep it >= 0.1
    return level,


def _minmax_level_to_arg(level, _hparams, min_val=0., max_val=1.0, clamp=True):
    level = (level / _LEVEL_DENOM)
    level = min_val + (max_val - min_val) * level
    if clamp:
        level = max(min_val, min(max_val, level))
    return level,


def _shear_level_to_arg(level, _hparams):
    # ra: range [-0.3, 0.3]; ta: range [-0.99, 0.99]
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        level_denom = _hparams.get("magnitude_max", 31)
        level = (level / level_denom) * 0.99
    else:
        level = (level / _LEVEL_DENOM) * 0.3
    level = _randomly_negate(level)
    return level,


def _translate_level_to_arg(level, _hparams):
    # default ra: range [-0.45, 0.45]; ta: 32/224
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        translate_pct = _hparams.get("translate_pct", 0.15)
        level_denom = _hparams.get("magnitude_max", 31)
        level = (level / level_denom) * translate_pct
    else:
        translate_pct = _hparams.get("translate_pct", 0.45)
        level = (level / _LEVEL_DENOM) * translate_pct
        level = _randomly_negate(level)
    return level,


def _posterize_level_to_arg(level, _hparams):
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        return _posterize_original_level_to_arg(level, _hparams)
    return int((level / _LEVEL_DENOM) * 4),


def _posterize_increasing_level_to_arg(level, hparams):
    return 4 - _posterize_level_to_arg(level, hparams)[0],


def _posterize_original_level_to_arg(level, _hparams):
    # According to the original AutoAugment paper instructions
    # range [4, 8], 'keep 4 up to 8 MSB of image'
    # augmented intensity/severity decreases with level
    # ta: range [2, 8]
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        level_denom = _hparams.get("magnitude_max", 31)
        return int((level / level_denom) * 6) + 2,
    return int((level / _LEVEL_DENOM) * 4) + 4,


def _solarize_level_to_arg(level, _hparams):
    # range [0, 255]
    # augmented intensity/severity decreases with level
    trivial_aug_wide = _hparams.get("trivialaugwide", False)
    if trivial_aug_wide:
        level_denom = _hparams.get("magnitude_max", 31)
        return int((level/level_denom)*255),
    return int((level / _LEVEL_DENOM) * 255),


def _solarize_increasing_level_to_arg(level, _hparams):
    # range [0, 255]
    # augmented intensity/severity increases with level
    return 255 - _solarize_level_to_arg(level, _hparams)[0],
# fmt: on


LEVEL_TO_ARG = {
    "AutoContrast": None,
    "Equalize": None,
    "Invert": None,
    "Rotate": _rotate_level_to_arg,
    "Posterize": _posterize_level_to_arg,
    "PosterizeIncreasing": _posterize_increasing_level_to_arg,
    "PosterizeOriginal": _posterize_original_level_to_arg,
    "Solarize": _solarize_level_to_arg,
    "SolarizeIncreasing": _solarize_increasing_level_to_arg,
    "Color": _enhance_level_to_arg,
    "ColorIncreasing": _enhance_increasing_level_to_arg,
    "Contrast": _enhance_level_to_arg,
    "ContrastIncreasing": _enhance_increasing_level_to_arg,
    "Brightness": _enhance_level_to_arg,
    "BrightnessIncreasing": _enhance_increasing_level_to_arg,
    "Sharpness": _enhance_level_to_arg,
    "SharpnessIncreasing": _enhance_increasing_level_to_arg,
    "ShearX": _shear_level_to_arg,
    "ShearY": _shear_level_to_arg,
    "TranslateX": _translate_level_to_arg,
    "TranslateY": _translate_level_to_arg,
    "Desaturate": partial(_minmax_level_to_arg, min_val=0.5, max_val=1.0),
    "GaussianBlurRand": _minmax_level_to_arg,
}

NAME_TO_OP = {
    "AutoContrast": auto_contrast,
    "Equalize": equalize,
    "Invert": invert,
    "Rotate": rotate,
    "Posterize": posterize,
    "PosterizeIncreasing": posterize,
    "PosterizeOriginal": posterize,
    "Solarize": solarize,
    "SolarizeIncreasing": solarize,
    "Color": color,
    "ColorIncreasing": color,
    "Contrast": contrast,
    "ContrastIncreasing": contrast,
    "Brightness": brightness,
    "BrightnessIncreasing": brightness,
    "Sharpness": sharpness,
    "SharpnessIncreasing": sharpness,
    "ShearX": shear_x,
    "ShearY": shear_y,
    "TranslateX": translate_x,
    "TranslateY": translate_y,
    "Desaturate": desaturate,
    "GaussianBlurRand": gaussian_blur_rand,
}


class AugmentOp:
    def __init__(self, name, prob=0.5, magnitude=10, hparams=None):
        hparams = hparams or _HPARAMS_DEFAULT
        self.name = name
        self.aug_fn = NAME_TO_OP[name]
        self.level_fn = LEVEL_TO_ARG[name]
        self.prob = prob
        self.magnitude = magnitude
        self.hparams = hparams.copy()
        self.kwargs = dict(
            fill_value=hparams["img_mean"] if "img_mean" in hparams else _FILL,
            resample=hparams["interpolation"] if "interpolation" in hparams else _RANDOM_INTERPOLATION,
        )

        # If magnitude_std is > 0, we introduce randomness into the usually fixed strategy
        # and sample magnitude from a normal distribution with mean `magnitude` and std-dev of `magnitude_std`.
        # If magnitude_std is inf, we sample magnitude from a uniform distribution.
        self.magnitude_std = self.hparams.get("magnitude_std", 0)
        self.magnitude_max = self.hparams.get("magnitude_max", None)

    def __call__(self, img):
        if self.prob < 1.0 and random.random() > self.prob:
            return img
        magnitude = self.magnitude
        if self.magnitude_std > 0:
            if self.magnitude_std == float("inf"):
                magnitude = random.uniform(0, magnitude)
            elif self.magnitude_std > 0:
                magnitude = random.gauss(magnitude, self.magnitude_std)
        upper_bound = self.magnitude_max or _LEVEL_DENOM
        magnitude = max(0.0, min(magnitude, upper_bound))
        level_args = self.level_fn(magnitude, self.hparams) if self.level_fn is not None else tuple()
        return self.aug_fn(img, *level_args, **self.kwargs)


def auto_augment_policy_posterize_original(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501
    policy = [
        [("PosterizeOriginal", 0.4, 8), ("Rotate", 0.6, 9)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeOriginal", 0.6, 7), ("PosterizeOriginal", 0.6, 6)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeOriginal", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeOriginal", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
        [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_posterize_increasing(hparams):
    # ImageNet policy from https://arxiv.org/abs/1805.09501 with research posterize variation
    policy = [
        [("PosterizeIncreasing", 0.4, 8), ("Rotate", 0.6, 9)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
        [("PosterizeIncreasing", 0.6, 7), ("PosterizeIncreasing", 0.6, 6)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Equalize", 0.4, 4), ("Rotate", 0.8, 8)],
        [("Solarize", 0.6, 3), ("Equalize", 0.6, 7)],
        [("PosterizeIncreasing", 0.8, 5), ("Equalize", 1.0, 2)],
        [("Rotate", 0.2, 3), ("Solarize", 0.6, 8)],
        [("Equalize", 0.6, 8), ("PosterizeIncreasing", 0.4, 6)],
        [("Rotate", 0.8, 8), ("Color", 0.4, 0)],
        [("Rotate", 0.4, 9), ("Equalize", 0.6, 2)],
        [("Equalize", 0.0, 7), ("Equalize", 0.8, 8)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Rotate", 0.8, 8), ("Color", 1.0, 2)],
        [("Color", 0.8, 8), ("Solarize", 0.8, 7)],
        [("Sharpness", 0.4, 7), ("Invert", 0.6, 8)],
        [("ShearX", 0.6, 5), ("Equalize", 1.0, 9)],
        [("Color", 0.4, 0), ("Equalize", 0.6, 3)],
        [("Equalize", 0.4, 7), ("Solarize", 0.2, 4)],
        [("Solarize", 0.6, 5), ("AutoContrast", 0.6, 5)],
        [("Invert", 0.6, 4), ("Equalize", 1.0, 8)],
        [("Color", 0.6, 4), ("Contrast", 1.0, 8)],
        [("Equalize", 0.8, 8), ("Equalize", 0.6, 3)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy_3a(hparams):
    policy = [
        [("Solarize", 1.0, 5)],  # 128 solarize threshold @ 5 magnitude
        [("Desaturate", 1.0, 10)],  # grayscale at 10 magnitude
        [("GaussianBlurRand", 1.0, 10)],
    ]
    pc = [[AugmentOp(*a, hparams=hparams) for a in sp] for sp in policy]
    return pc


def auto_augment_policy(name="autoaug", hparams=None):
    hparams = hparams or _HPARAMS_DEFAULT
    if name == "autoaug":
        return auto_augment_policy_posterize_original(hparams)
    elif name == "autoaugr":
        return auto_augment_policy_posterize_increasing(hparams)
    elif name == "3a":
        return auto_augment_policy_3a(hparams)
    else:
        assert False, "Unknown auto augment policy (%s)" % name


class AutoAugment:
    def __init__(self, policy):
        self.policy = policy

    def __call__(self, img):
        sub_policy = random.choice(self.policy)
        for op in sub_policy:
            img = op(img)
        return img


def auto_augment_transform(configs, hparams):
    """
    Create a AutoAugment transform
    Args:
        configs: A string that defines the automatic augmentation configuration.
            It is composed of multiple parts separated by dashes ("-"). The first part defines
            the AutoAugment policy ('autoaug', 'autoaugr' or '3a':
            'autoaug' for the original AutoAugment policy with PosterizeOriginal,
            'autoaugr' for the AutoAugment policy with PosterizeIncreasing operation,
             '3a' for the AutoAugment only with 3 augmentations.)
            There is no order requirement for the remaining config parts.

            - mstd: Float standard deviation of applied magnitude noise.

            Example: 'autoaug-mstd0.5' will be automatically augment using the autoaug strategy
            and magnitude_std 0.5.
        hparams: Other hparams of the automatic augmentation scheme.
    """
    config = configs.split("-")
    policy_name = config[0]
    config = config[1:]
    hparams.setdefault("magnitude_std", 0.5)  # default magnitude_std is set to 0.5
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param injected via hparams for now
            hparams.setdefault("magnitude_std", float(val))
        else:
            assert False, "Unknown AutoAugment config section"
    aa_policy = auto_augment_policy(policy_name, hparams=hparams)
    return AutoAugment(aa_policy)


_RAND_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "Posterize",
    "Solarize",
    "Color",
    "Contrast",
    "Brightness",
    "Sharpness",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]

_RAND_INCREASING_TRANSFORMS = [
    "AutoContrast",
    "Equalize",
    "Invert",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "ColorIncreasing",
    "ContrastIncreasing",
    "BrightnessIncreasing",
    "SharpnessIncreasing",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]

# These experimental weights are roughly based on the relative improvements mentioned in the paper.
# They may not result in increased performance, but could likely be tuned to so.
_RAND_CHOICE_WEIGHTS_0 = {
    "Rotate": 0.3,
    "ShearX": 0.2,
    "ShearY": 0.2,
    "TranslateX": 0.1,
    "TranslateY": 0.1,
    "Color": 0.025,
    "Sharpness": 0.025,
    "AutoContrast": 0.025,
    "Solarize": 0.005,
    "Contrast": 0.005,
    "Brightness": 0.005,
    "Equalize": 0.005,
    "Posterize": 0.005,
    "Invert": 0,
}


def _select_rand_weights(weight_idx=0, transforms=None):
    transforms = transforms or _RAND_TRANSFORMS
    assert weight_idx == 0  # only one set of weights currently
    rand_weights = _RAND_CHOICE_WEIGHTS_0
    probs = [rand_weights[k] for k in transforms]
    probs /= np.sum(probs)
    return probs


def rand_augment_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(name, prob=0.5, magnitude=magnitude, hparams=hparams) for name in transforms]


class RandAugment:
    def __init__(self, ops, num_layers=2, choice_weights=None):
        self.ops = ops
        self.num_layers = num_layers
        self.choice_weights = choice_weights

    def __call__(self, img):
        # no replacement when using weighted choice
        ops = np.random.choice(self.ops, self.num_layers, replace=self.choice_weights is None, p=self.choice_weights)
        for op in ops:
            img = op(img)
        return img


def rand_augment_transform(configs, hparams):
    """
    Create a RandAugment transform
    Args:
        configs: A string that defines the random augmentation configuration.
            It is composed of multiple parts separated by dashes ("-").
            The first part defines the AutoAugment policy ('randaug' policy).
            There is no order requirement for the remaining config parts.

            - m: Integer magnitude of rand augment. Default: 10
            - n: Integer num layer (number of transform operations selected for each image). Default: 2
            - w: Integer probability weight index (the index that affects a group of weights selected by operations).
            - mstd: Floating standard deviation of applied magnitude noise,
                or uniform sampling at infinity (or greater than 100).
            - mmax: Set the upper range limit for magnitude to a value
                other than the default value of _LEVEL_DENOM (10).
            - inc: Integer (bool), using the severity increase with magnitude (default: 0).

            Example: 'randaug-w0-n3-mstd0.5' will be random augment
                using the weights 0, num_layers 3, magnitude_std 0.5.
        hparams: Other hparams (kwargs) for the RandAugmentation scheme.
    """
    magnitude = _LEVEL_DENOM  # default to _LEVEL_DENOM for magnitude (currently 10)
    num_layers = 2  # default to 2 ops per image
    hparams.setdefault("magnitude_std", 0.5)  # default magnitude_std is set to 0.5
    weight_idx = None  # default to no probability weights for op choice
    transforms = _RAND_TRANSFORMS
    config = configs.split("-")
    assert config[0] == "randaug"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "mstd":
            # noise param / randomization of magnitude values
            mstd = float(val)
            if mstd > 100:
                # use uniform sampling in 0 to magnitude if mstd is > 100
                mstd = float("inf")
            hparams.setdefault("magnitude_std", mstd)
        elif key == "mmax":
            # clip magnitude between [0, mmax] instead of default [0, _LEVEL_DENOM]
            hparams.setdefault("magnitude_max", int(val))
        elif key == "inc":
            if bool(val):
                transforms = _RAND_INCREASING_TRANSFORMS
        elif key == "m":
            magnitude = int(val)
        elif key == "n":
            num_layers = int(val)
        elif key == "w":
            weight_idx = int(val)
        else:
            assert False, "Unknown RandAugment config section"
    ra_ops = rand_augment_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    choice_weights = None if weight_idx is None else _select_rand_weights(weight_idx)
    return RandAugment(ra_ops, num_layers, choice_weights=choice_weights)


_TRIVIALAUGMENT_WIDE_TRANSFORMS = _RAND_TRANSFORMS


def trivial_augment_wide_ops(magnitude=31, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _RAND_TRANSFORMS
    return [AugmentOp(name, prob=1.0, magnitude=magnitude, hparams=hparams) for name in transforms]


class TrivialAugmentWide:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, img):
        op = np.random.choice(self.ops)
        img = op(img)
        return img


def trivial_augment_wide_transform(configs, hparams):
    """
    Create a TrivialAugmentWide transform
    Args:
        configs: A string that defines the TrivialAugmentWide configuration.
            It is composed of multiple parts separated by dashes ("-").
            The first part defines the AutoAugment name, it should be 'trivialaugwide'.
            the second part(not necessary) the maximum value of magnitude.

            - m: final magnitude of a operation will uniform sampling from [0, m] . Default: 31

            Example: 'trivialaugwide-m20' will be TrivialAugmentWide
            with mgnitude uniform sampling from [0, 20],
        hparams: Other hparams (kwargs) for the TrivialAugment scheme.
    Returns:
        A Mindspore compatible Transform
    """
    magnitude = 31
    transforms = _TRIVIALAUGMENT_WIDE_TRANSFORMS
    config = configs.split("-")
    assert config[0] == "trivialaugwide"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "m":
            magnitude = int(val)
        else:
            assert False, "Unknown TrivialAugmentWide config section"
    if not hparams:
        hparams = dict()
    hparams["magnitude_max"] = magnitude
    hparams["magnitude_std"] = float("inf")  # default to uniform sampling
    hparams["trivialaugwide"] = True
    ta_ops = trivial_augment_wide_ops(magnitude=magnitude, hparams=hparams, transforms=transforms)
    return TrivialAugmentWide(ta_ops)


_AUGMIX_TRANSFORMS = [
    "AutoContrast",
    "ColorIncreasing",  # not in paper
    "ContrastIncreasing",  # not in paper
    "BrightnessIncreasing",  # not in paper
    "SharpnessIncreasing",  # not in paper
    "Equalize",
    "Rotate",
    "PosterizeIncreasing",
    "SolarizeIncreasing",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]


def augmix_ops(magnitude=10, hparams=None, transforms=None):
    hparams = hparams or _HPARAMS_DEFAULT
    transforms = transforms or _AUGMIX_TRANSFORMS
    return [AugmentOp(name, prob=1.0, magnitude=magnitude, hparams=hparams) for name in transforms]


class AugMixAugment:
    """AugMix Transform
    according the  paper: "AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    """

    def __init__(self, ops, alpha=1.0, width=3, depth=-1):
        self.ops = ops
        self.alpha = alpha
        self.width = width
        self.depth = depth

    def __call__(self, img):
        mixing_weights = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        mixed = np.zeros(img.shape, dtype=np.float32)
        for mw in mixing_weights:
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            ops = np.random.choice(self.ops, depth, replace=True)
            img_aug = img  # no ops are in-place, deep copy not necessary
            for op in ops:
                img_aug = op(img_aug)
            mixed += mw * img_aug.astype(np.float32)
        np.clip(mixed, 0, 255.0, out=mixed)
        mixed = mixed.astype(np.uint8)
        img = img * (1 - m) + mixed * m
        return img


def augment_and_mix_transform(configs, hparams=None):
    """Create AugMix PyTorch transform

    Args:
        configs (str): String defining configuration of AugMix augmentation. Consists of multiple sections separated
            by dashes ('-'). The first section defines the specific name of augment, it should be 'augmix'.
            The remaining sections, not order sepecific determine
                'm' - integer magnitude (severity) of augmentation mix (default: 3)
                'w' - integer width of augmentation chain (default: 3)
                'd' - integer depth of augmentation chain (-1 is random [1, 3], default: -1)
                'a' - integer or float, the args of beta deviation of beta for generate the weight, default 1..
            Ex 'augmix-m5-w4-d2' results in AugMix with severity 5, chain width 4, chain depth 2

        hparams: Other hparams (kwargs) for the Augmentation transforms

    Returns:
         A Mindspore compatible Transform
    """
    magnitude = 3
    width = 3
    depth = -1
    alpha = 1.0
    config = configs.split("-")
    assert config[0] == "augmix"
    config = config[1:]
    for c in config:
        cs = re.split(r"(\d.*)", c)
        if len(cs) < 2:
            continue
        key, val = cs[:2]
        if key == "m":
            magnitude = int(val)
        elif key == "w":
            width = int(val)
        elif key == "d":
            depth = int(val)
        elif key == "a":
            alpha = float(val)
        else:
            assert False, "Unknown AugMix config section"
    if not hparams:
        hparams = dict()
    hparams["magnitude_std"] = float("inf")  # default to uniform sampling (if not set via mstd arg)
    ops = augmix_ops(magnitude=magnitude, hparams=hparams)
    return AugMixAugment(ops, alpha=alpha, width=width, depth=depth)
