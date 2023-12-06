"""
transform for text recognition tasks.
"""
import logging
import math
from random import sample
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

__all__ = [
    "RecCTCLabelEncode",
    "RecAttnLabelEncode",
    "RecMasterLabelEncode",
    "VisionLANLabelEncode",
    "RecResizeImg",
    "RecResizeNormForInfer",
    "SVTRRecResizeImg",
    "Rotate90IfVertical",
    "ClsLabelEncode",
    "SARLabelEncode",
    "RobustScannerRecResizeImg",
]
_logger = logging.getLogger(__name__)


class RecCTCLabelEncode(object):
    """Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:
        max_text_len: to pad the label text to a fixed length (max_text_len) of text for ctc loss computate.
        character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
            (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
        use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
        blank_at_last(bool): padding with blank index (not the space index). If True, a blank/padding token will be
            appended to the end of the dictionary, so that blank_index = num_chars, where num_chars is the number of
            character in the dictionary including space char if used. If False, blank token will be inserted in the
            beginning of the dictionary, so blank_index=0.
        lower (bool): if True, all upper-case chars in the label text will be converted to lower case.
            Set to be True if dictionary only contains lower-case chars.
            Set to be False if not and want to recognition both upper-case and lower-case.

    Attributes:
        blank_idx: the index of the blank token for padding
        num_valid_chars: the number of valid characters (including space char if used) in the dictionary
        num_classes: the number of classes (which valid characters char and the speical token for blank padding).
            so num_classes = num_valid_chars + 1


    """

    def __init__(
        self,
        max_text_len=23,
        character_dict_path=None,
        use_space_char=False,
        blank_at_last=True,
        lower=False,
        **kwargs,
        # start_token='<BOS>',
        # end_token='<EOS>',
        # unkown_token='',
    ):
        self.max_text_len = max_text_len
        self.space_idx = None
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = [c for c in "0123456789abcdefghijklmnopqrstuvwxyz"]

            self.lower = True
        else:
            # TODO: this is commonly used in other modules, wrap into a func or class.
            # parse char dictionary
            char_list = []
            with open(character_dict_path, "r") as f:
                for line in f:
                    c = line.rstrip("\n\r")
                    char_list.append(c)
        # add space char if set
        if use_space_char:
            if " " not in char_list:
                char_list.append(" ")
            self.space_idx = len(char_list) - 1
        else:
            if " " in char_list:
                _logger.warning(
                    "The dict still contains space char in dict although use_space_char is set to be False, "
                    f"because the space char is coded in the dictionary file {character_dict_path}"
                )

        self.num_valid_chars = len(char_list)  # the number of valid chars (including space char if used)

        # add blank token for padding
        if blank_at_last:
            # the index of a char in dict is [0, num_chars-1], blank index is set to num_chars
            char_list.append("<PAD>")
            self.blank_idx = self.num_valid_chars
        else:
            char_list = ["<PAD>"] + char_list
            self.blank_idx = 0

        self.dict = {c: idx for idx, c in enumerate(char_list)}

        self.num_classes = len(self.dict)

    def __call__(self, data: dict):
        """
        required keys:
            label -> (str) text string
        added keys:
            text_seq-> (np.ndarray, int32), sequence of character indices after  padding to max_text_len in shape
            (sequence_len), where ood characters are skipped
        added keys:
            length -> (np.int32) the number of valid chars in the encoded char index sequence,  where valid means
            the char is in dictionary.
            text_padded -> (str) text label padded to fixed length, to solved the dynamic shape issue in dataloader.
            text_length -> int, the length of original text string label

        """
        char_indices = str2idx(data["label"], self.dict, max_text_len=self.max_text_len, lower=self.lower)

        if char_indices is None:
            char_indices = []
            # return None
        data["length"] = np.array(len(char_indices), dtype=np.int32)
        # padding with blank index
        char_indices = char_indices + [self.blank_idx] * (self.max_text_len - len(char_indices))
        # TODO: raname to char_indices
        data["text_seq"] = np.array(char_indices, dtype=np.int32)
        #
        data["text_length"] = len(data["label"])
        data["text_padded"] = data["label"] + " " * (self.max_text_len - len(data["label"]))

        return data


class VisionLANLabelEncode(RecCTCLabelEncode):
    """Convert text label (str) to the labels needed by the VisionLAN, inheritated from RecCTCLabelEncode

    Args:
        max_text_len: a fixed length (max_text_len) of char indices, which the label, label_res, label_mas,
            text_padded will be padded to.
        character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
            (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
        use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
        blank_at_last(bool): padding with blank index (not the space index). If True, a blank/padding token will be
            appended to the end of the dictionary, so that blank_index = num_chars, where num_chars is the number of
            character in the dictionary including space char if used. If False, blank token will be inserted in the
            beginning of the dictionary, so blank_index=0.
        lower (bool): if True, all upper-case chars in the label text will be converted to lower case.
            Set to be True if dictionary only contains lower-case chars.
            Set to be False if not and want to recognition both upper-case and lower-case.

    Attributes:
        blank_idx: the index of the blank token for padding
        max_text_len: the padded text length
        num_valid_chars: the number of valid characters (including space char if used) in the dictionary
        num_classes: the number of classes (which valid characters char and the speical token for blank padding).
            so num_classes = num_valid_chars + 1
    """

    def __init__(
        self, max_text_len, character_dict_path=None, use_space_char=False, blank_at_last=True, lower=False, **kwargs
    ):
        super(VisionLANLabelEncode, self).__init__(
            max_text_len, character_dict_path, use_space_char, blank_at_last, lower
        )
        assert (
            not blank_at_last
        ), "VisionLAN applies the blank token at the beginning of the dictionary, so the blank_at_last should be False"
        self.max_text_len = self.max_text_len + 1  # since VisionLAN predicts EOS, increaset the max_text_len by 1

    def __call__(self, data):
        """
        required keys:
            label -> (str) original text string
        added keys:
            label_id -> (int), the index for the randomly chosen character to be occluded
            label -> (np.ndarray),  sequence of character indices for the original text
                                    string after padding to max_text_len
            label_res -> (np.ndarray), sequence of character indices where the character is
                                    removed after padding to max_text_len
            label_sub -> (np.ndarray),  sequence of character indices of the occluded character
                                    after padding to max_text_len
            length -> (np.int32) the number of valid chars in the encoded char index sequence,
                                    where valid means the char is in dictionary.
            text_padded ->  text string padded to fixed length, to solved the dynamic shape
                                    issue in dataloader.
        """
        text = data["label"]  # original string
        # 1. randomly select a character to be occluded, save its index to label_id
        len_str = len(text)
        if len_str == 0:
            raise ValueError("The length of the label string is zero")
        change_num = 1
        order = list(range(len_str))
        label_id = sample(order, change_num)[0]  # randomly select the change character index
        # 2. obtain two strings: label_sub and label_res
        label_sub = text[label_id]
        if label_id == (len_str - 1):
            label_res = text[:label_id]
        elif label_id == 0:
            label_res = text[1:]
        else:
            label_res = text[:label_id] + text[label_id + 1 :]

        data["label_id"] = label_id  # character index
        # 3. encode strings (valid characters) to indices
        char_indices = str2idx(data["label"], self.dict, max_text_len=self.max_text_len, lower=self.lower)
        if char_indices is None:
            char_indices = []
        label_res = str2idx(label_res, self.dict, max_text_len=self.max_text_len, lower=self.lower, ignore_warning=True)
        label_sub = str2idx(label_sub, self.dict, max_text_len=self.max_text_len, lower=self.lower, ignore_warning=True)
        if label_res is None:
            label_res = []
        if label_sub is None:
            label_sub = []
        data["length"] = len(char_indices)
        # 4. pad to a fixed length by appending zeros (self.blank_idx)
        char_indices = char_indices + [self.blank_idx] * (self.max_text_len - len(char_indices))
        data["text_padded"] = data["label"] + " " * (self.max_text_len - len(data["label"]))
        data["label"] = np.array(char_indices)
        label_res = label_res + [self.blank_idx] * (self.max_text_len - len(label_res))
        label_sub = label_sub + [self.blank_idx] * (self.max_text_len - len(label_sub))
        data["label_res"] = np.array(label_res)
        data["label_sub"] = np.array(label_sub)
        return data


class RecAttnLabelEncode:
    def __init__(
        self,
        max_text_len: int = 25,
        character_dict_path: Optional[str] = None,
        use_space_char: bool = False,
        lower: bool = False,
        **kwargs,
    ) -> None:
        """
        Convert text label (str) to a sequence of character indices according to the char dictionary

        Args:
            max_text_len: to pad the label text to a fixed length (max_text_len) of text for attn loss computate.
            character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
                (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
            use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
            lower (bool): if True, all upper-case chars in the label text will be converted to lower case.
                Set to be True if dictionary only contains lower-case chars. Set to be False if not and want to
                recognition both upper-case and lower-case.

        Attributes:
            go_idx: the index of the GO token
            stop_idx: the index of the STOP token
            num_valid_chars: the number of valid characters (including space char if used) in the dictionary
            num_classes: the number of classes (which valid characters char and the speical token for blank padding).
            so num_classes = num_valid_chars + 1
        """
        self.max_text_len = max_text_len
        self.space_idx = None
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = list("0123456789abcdefghijklmnopqrstuvwxyz")

            self.lower = True
            _logger.info("The character_dict_path is None, model can only recognize number and lower letters")
        else:
            # parse char dictionary
            char_list = []
            with open(character_dict_path, "r") as f:
                for line in f:
                    c = line.rstrip("\n\r")
                    char_list.append(c)

        # add space char if set
        if use_space_char:
            if " " not in char_list:
                char_list.append(" ")
            self.space_idx = len(char_list) - 1
        else:
            if " " in char_list:
                _logger.warning(
                    "The dict still contains space char in dict although use_space_char is set to be False, "
                    f"because the space char is coded in the dictionary file {character_dict_path}"
                )

        self.num_valid_chars = len(char_list)  # the number of valid chars (including space char if used)

        special_token = ["<GO>", "<STOP>"]
        char_list = special_token + char_list

        self.go_idx = 0
        self.stop_idx = 1

        self.dict = {c: idx for idx, c in enumerate(char_list)}

        self.num_classes = len(self.dict)

    def __call__(self, data: Dict[str, Any]) -> str:
        char_indices = str2idx(data["label"], self.dict, max_text_len=self.max_text_len, lower=self.lower)

        if char_indices is None:
            char_indices = []
        data["length"] = np.array(len(char_indices), dtype=np.int32)

        char_indices = (
            [self.go_idx] + char_indices + [self.stop_idx] + [self.go_idx] * (self.max_text_len - len(char_indices))
        )
        data["text_seq"] = np.array(char_indices, dtype=np.int32)

        data["text_length"] = len(data["label"])
        data["text_padded"] = data["label"] + " " * (self.max_text_len - len(data["label"]))
        return data


class RecMasterLabelEncode:
    def __init__(
        self,
        max_text_len: int = 25,
        character_dict_path: Optional[str] = None,
        use_space_char: bool = False,
        use_unknown_char: bool = False,
        lower: bool = False,
        **kwargs,
    ) -> None:
        """
        Convert text label (str) to a sequence of character indices according to the char dictionary

        Args:
            max_text_len: to pad the label text to a fixed length (max_text_len) of text for attn loss computate.
            character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
                (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
            use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
            use_unknown_char(bool): Use the unknown character to replace the unknown character instead of skipping
            lower (bool): if True, all upper-case chars in the label text will be converted to lower case.
                Set to be True if dictionary only contains lower-case chars. Set to be False if not and want to
                recognition both upper-case and lower-case.

        Attributes:
            go_idx: the index of the GO token
            stop_idx: the index of the STOP token
            pad_idx: the index of the PAD token
            num_valid_chars: the number of valid characters (including space char if used) in the dictionary
            num_classes: the number of classes (which valid characters char and the speical token for blank padding).
                so num_classes = num_valid_chars + 1
        """
        self.max_text_len = max_text_len
        self.space_idx = None
        self.unknown_idx = None
        self.unknown_token = "<UNKNOWN>"
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = list("0123456789abcdefghijklmnopqrstuvwxyz")

            self.lower = True
            _logger.info("The character_dict_path is None, model can only recognize number and lower letters")
        else:
            # parse char dictionary
            char_list = []
            with open(character_dict_path, "r") as f:
                for line in f:
                    c = line.rstrip("\n\r")
                    char_list.append(c)

        # add space char if set
        if use_space_char:
            if " " not in char_list:
                char_list.append(" ")
            self.space_idx = len(char_list) - 1
        else:
            if " " in char_list:
                _logger.warning(
                    "The dict still contains space char in dict although use_space_char is set to be False, "
                    f"because the space char is coded in the dictionary file {character_dict_path}"
                )

        self.num_valid_chars = len(char_list)  # the number of valid chars (including space char if used)

        special_token = ["<GO>", "<STOP>", "<PAD>"]
        char_list = special_token + char_list

        self.go_idx = 0
        self.stop_idx = 1
        self.pad_idx = 2

        # use unknow char if set
        if use_unknown_char:
            char_list = char_list + [self.unknown_token]
            self.unknown_idx = len(char_list) - 1

        self.dict = {c: idx for idx, c in enumerate(char_list)}

        self.num_classes = len(self.dict)

    def __call__(self, data: Dict[str, Any]) -> str:
        char_indices = str2idx(
            data["label"], self.dict, max_text_len=self.max_text_len, lower=self.lower, unknown_idx=self.unknown_idx
        )

        if char_indices is None:
            char_indices = []
        data["length"] = np.array(len(char_indices), dtype=np.int32)

        char_indices = (
            [self.go_idx] + char_indices + [self.stop_idx] + [self.pad_idx] * (self.max_text_len - len(char_indices))
        )
        data["text_seq"] = np.array(char_indices, dtype=np.int32)

        data["text_length"] = len(data["label"])
        data["text_padded"] = data["label"] + " " * (self.max_text_len - len(data["label"]))
        return data


def str2idx(
    text: str,
    label_dict: Dict[str, int],
    max_text_len: int = 23,
    lower: bool = False,
    unknown_idx: Optional[int] = None,
    ignore_warning: bool = False,
) -> List[int]:
    """
    Encode text (string) to a squence of char indices
    Args:
        text (str): text string
    Returns:
        char_indices (List[int]): char index seq
    """
    if len(text) == 0 or len(text) > max_text_len:
        return None

    if lower:
        text = text.lower()

    char_indices = []
    for char in text:
        if char not in label_dict:
            if unknown_idx is not None:
                char_indices.append(unknown_idx)
        else:
            char_indices.append(label_dict[char])

    if len(char_indices) == 0 and not ignore_warning:
        _logger.warning("`{}` does not contain any valid character in the dictionary.".format(text))
        return None

    return char_indices


# TODO: reorganize the code for different resize transformation in rec task
def resize_norm_img(img, image_shape, padding=True, interpolation=cv2.INTER_LINEAR):
    """
    resize image
    Args:
        img: shape (H, W, C)
        image_shape: image shape after resize, in (C, H, W)
        padding: if Ture, resize while preserving the H/W ratio, then pad the blank.

    """
    imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
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

    padding_im = np.zeros((imgH, imgW, c), dtype=resized_image.dtype)
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


# TODO: check diff from resize_norm_img
def resize_norm_img_chinese(img, image_shape):
    """adopted from paddle"""
    imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    c = img.shape[2]
    ratio = w * 1.0 / h

    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))

    padding_im = np.zeros((imgH, imgW, c), dtype=resized_image.dtype)
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio


# TODO: remove infer_mode and character_dict_path if they are not necesary
class RecResizeImg(object):
    """adopted from paddle
    resize, convert from hwc to chw, rescale pixel value to -1 to 1
    """

    def __init__(self, image_shape, infer_mode=False, character_dict_path=None, padding=True, **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data["image"]
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape, self.padding)
        data["image"] = norm_img
        data["valid_ratio"] = valid_ratio
        # TODO: data['shape_list'] = ?
        return data


class SVTRRecResizeImg(object):
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, data):
        img = data["image"]

        norm_img, valid_ratio = resize_norm_img(img, self.image_shape, self.padding)
        data["image"] = norm_img
        data["valid_ratio"] = valid_ratio
        return data


class RecResizeNormForInfer(object):
    """
    Resize image for text recognition

    Args:
        target_height: target height after resize. Commonly, 32 for crnn, 48 for svtr. default is 32.
        target_width: target width. Default is 320. If None, image width is scaled to make aspect ratio unchanged.
        keep_ratio: keep aspect ratio.
            If True, resize the image with ratio=target_height / input_height (certain image height is required by
            recognition network).
            If False, simply resize to targte size (`target_height`, `target_width`)
        padding: If True, pad the resized image to the targte size with zero RGB values.
            only used when `keep_ratio` is True.

    Notes:
        1. The default choice (keep_ratio, not padding) is suitable for inference for better accuracy.
    """

    def __init__(
        self,
        target_height=32,
        target_width=320,
        keep_ratio=True,
        padding=False,
        interpolation=cv2.INTER_LINEAR,
        norm_before_pad=False,
        mean=[127.0, 127.0, 127.0],
        std=[127.0, 127.0, 127.0],
        divisor=None,
        **kwargs,
    ):
        self.keep_ratio = keep_ratio
        self.padding = padding
        # self.targt_shape = target_shape
        self.tar_h = target_height
        self.tar_w = target_width
        self.interpolation = interpolation
        self.norm_before_pad = norm_before_pad
        self.mean = np.array(mean, dtype="float32")
        self.std = np.array(std, dtype="float32")
        self.divisor = divisor

    def norm(self, img):
        if self.divisor:
            img = img / self.divisor
        return (img - self.mean) / self.std

    def __call__(self, data):
        """
        data: image in shape [h, w, c]
        """
        img = data["image"]
        h, w = img.shape[:2]
        # tar_h, tar_w = self.targt_shape
        resize_h = self.tar_h

        if "max_wh_ratio" in data:
            max_wh_ratio = data["max_wh_ratio"]
        else:
            max_wh_ratio = self.tar_w / float(self.tar_h)

        img_w = int(resize_h * max_wh_ratio)
        if not self.keep_ratio:
            assert self.tar_w is not None, "Must specify target_width if keep_ratio is False"
            resize_w = self.tar_w  # if self.tar_w is not None else resized_h * self.max_wh_ratio
        else:
            src_wh_ratio = w / float(h)
            resize_w = img_w if img_w < math.ceil(resize_h * src_wh_ratio) else int(math.ceil(resize_h * src_wh_ratio))
        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interpolation).astype("float32")

        # TODO: norm before padding

        data["shape_list"] = np.array(
            [h, w, resize_h / h, resize_w / w], dtype=np.float32
        )  # TODO: reformat, currently align to det
        if self.norm_before_pad:
            resized_img = self.norm(resized_img)

        if self.padding and self.keep_ratio:
            padded_img = np.zeros((self.tar_h, img_w, 3), dtype=resized_img.dtype)
            padded_img[:, :resize_w, :] = resized_img
            data["image"] = padded_img
        else:
            data["image"] = resized_img

        if not self.norm_before_pad:
            data["image"] = self.norm(data["image"])

        return data


class Rotate90IfVertical:
    """Rotate the image by 90 degree when the height/width ratio is larger than the given threshold.
    Note: It needs to be called before image resize."""

    def __init__(self, threshold: float = 1.5, direction: str = "counterclockwise", **kwargs):
        self.threshold = threshold

        if direction == "counterclockwise":
            self.flag = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif direction == "clockwise":
            self.flag = cv2.ROTATE_90_CLOCKWISE
        else:
            raise ValueError("Unsupported direction")

    def __call__(self, data):
        img = data["image"]

        h, w, _ = img.shape
        if h / w > self.threshold:
            img = cv2.rotate(img, self.flag)

        data["image"] = img
        return data


class ClsLabelEncode(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data["label"]
        assert (
            label in self.label_list
        ), f"Invalid label `{label}`. Please make sure each label in input data is one of the values \
            in label_list `{self.label_list}` in yaml config file."
        label = self.label_list.index(label)
        data["label"] = label

        return data


class SARLabelEncode(object):
    """Convert between text-label and text-index"""

    def __init__(
        self, max_text_len, character_dict_path=None, use_space_char=False, lower=False, is_training=True, **kwargs
    ):
        self.max_text_len = max_text_len
        self.beg_str = "sos"
        self.end_str = "eos"
        self.lower = lower
        self.is_training = is_training

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
            self.lower = True
            if self.is_training:
                _logger.warning("The character_dict_path is None, model can only recognize number and lower letters")
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
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list

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
        text_str = text
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len - 1:
            return None
        data["text_length"] = np.array(len(text))
        target = [self.start_idx] + text + [self.end_idx]
        padded_text = [self.padding_idx for _ in range(self.max_text_len)]

        padded_text[: len(target)] = target
        data["label"] = np.array(padded_text)
        data["text_padded"] = text_str + " " * (self.max_text_len - len(text_str))

        return data

    def get_ignored_tokens(self):
        return [self.padding_idx]


class RobustScannerRecResizeImg(object):
    def __init__(self, image_shape, max_text_len, width_downsample_ratio=0.25, **kwargs):
        self.image_shape = image_shape
        self.width_downsample_ratio = width_downsample_ratio
        self.max_text_len = max_text_len

    def __call__(self, data):
        img = data["image"]
        norm_img, resize_shape, pad_shape, valid_ratio = resize_norm_img_sar(
            img, self.image_shape, self.width_downsample_ratio
        )
        valid_ratio = np.array(valid_ratio, dtype=np.float32)
        width_downsampled = int(self.image_shape[-1] * self.width_downsample_ratio)
        valid_width_mask = np.full([1, width_downsampled], 1).astype("int32")
        valid_width = min(width_downsampled, int(width_downsampled * valid_ratio + 0.5))
        valid_width_mask[:, valid_width:] = 0
        word_positons = np.array(range(0, self.max_text_len)).astype("int32")
        data["image"] = norm_img
        data["resized_shape"] = resize_shape
        data["pad_shape"] = pad_shape
        data["valid_ratio"] = valid_ratio
        data["valid_width_mask"] = valid_width_mask
        data["word_positions"] = word_positons
        return data


def resize_norm_img_sar(img, image_shape, width_downsample_ratio=0.25):
    imgC, imgH, imgW_min, imgW_max = image_shape
    h = img.shape[0]
    w = img.shape[1]
    valid_ratio = 1.0
    # make sure new_width is an integral multiple of width_divisor.
    width_divisor = int(1 / width_downsample_ratio)
    # resize
    ratio = w / float(h)
    resize_w = math.ceil(imgH * ratio)
    if resize_w % width_divisor != 0:
        resize_w = round(resize_w / width_divisor) * width_divisor
    if imgW_min is not None:
        resize_w = max(imgW_min, resize_w)
    if imgW_max is not None:
        valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
        resize_w = min(imgW_max, resize_w)
    resized_image = cv2.resize(img, (resize_w, imgH))
    resized_image = resized_image.astype("float32")
    # norm
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    resize_shape = resized_image.shape
    padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
    padding_im[:, :, 0:resize_w] = resized_image
    pad_shape = padding_im.shape

    return padding_im, resize_shape, pad_shape, valid_ratio
