import logging

import cv2
import numpy as np
from PIL import Image

_logger = logging.getLogger(__name__)


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


class AttnLabelEncode(BaseRecLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(self, max_text_length, character_dict_path=None, use_space_char=False, **kwargs):
        super(AttnLabelEncode, self).__init__(max_text_length, character_dict_path, use_space_char)

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data["length"] = np.array(len(text))
        text = [0] + text + [len(self.character) - 1] + [0] * (self.max_text_len - len(text) - 2)
        data["label"] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, "Unsupport type %s in get_beg_end_flag_idx" % beg_or_end
        return idx


class TableLabelEncode(AttnLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        max_text_length,
        character_dict_path,
        replace_empty_cell_token=False,
        merge_no_span_structure=False,
        learn_empty_box=False,
        loc_reg_num=4,
        **kwargs
    ):
        self.max_text_len = max_text_length
        self.lower = False
        self.learn_empty_box = learn_empty_box
        self.merge_no_span_structure = merge_no_span_structure
        self.replace_empty_cell_token = replace_empty_cell_token

        dict_character = []
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode("utf-8").strip("\n").strip("\r\n")
                dict_character.append(line)

        if self.merge_no_span_structure:
            if "<td></td>" not in dict_character:
                dict_character.append("<td></td>")
            if "<td>" in dict_character:
                dict_character.remove("<td>")

        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.idx2char = {v: k for k, v in self.dict.items()}

        self.character = dict_character
        self.loc_reg_num = loc_reg_num
        self.pad_idx = self.dict[self.beg_str]
        self.start_idx = self.dict[self.beg_str]
        self.end_idx = self.dict[self.end_str]

        self.td_token = ["<td>", "<td", "<eb></eb>", "<td></td>"]
        self.empty_bbox_token_dict = {
            "[]": "<eb></eb>",
            "[' ']": "<eb1></eb1>",
            "['<b>', ' ', '</b>']": "<eb2></eb2>",
            "['\\u2028', '\\u2028']": "<eb3></eb3>",
            "['<sup>', ' ', '</sup>']": "<eb4></eb4>",
            "['<b>', '</b>']": "<eb5></eb5>",
            "['<i>', ' ', '</i>']": "<eb6></eb6>",
            "['<b>', '<i>', '</i>', '</b>']": "<eb7></eb7>",
            "['<b>', '<i>', ' ', '</i>', '</b>']": "<eb8></eb8>",
            "['<i>', '</i>']": "<eb9></eb9>",
            "['<b>', ' ', '\\u2028', ' ', '\\u2028', ' ', '</b>']": "<eb10></eb10>",
        }

    @property
    def _max_text_len(self):
        return self.max_text_len + 2

    def __call__(self, data):
        cells = data["cells"]
        structure = data["structure"]
        if self.merge_no_span_structure:
            structure = self._merge_no_span_structure(structure)
        if self.replace_empty_cell_token:
            structure = self._replace_empty_cell_token(structure, cells)
        # remove empty token and add " " to span token
        new_structure = []
        for token in structure:
            if token != "":
                if "span" in token and token[0] != " ":
                    token = " " + token
                new_structure.append(token)
        # encode structure
        structure = self.encode(new_structure)
        if structure is None:
            return None

        structure = [self.start_idx] + structure + [self.end_idx]  # add sos abd eos
        structure = structure + [self.pad_idx] * (self._max_text_len - len(structure))  # pad
        structure = np.array(structure)
        data["structure"] = structure

        if len(structure) > self._max_text_len:
            return None

        # encode box
        bboxes = np.zeros((self._max_text_len, self.loc_reg_num), dtype=np.float32)
        bbox_masks = np.zeros((self._max_text_len, 1), dtype=np.float32)

        bbox_idx = 0

        for i, token in enumerate(structure):
            if self.idx2char[token] in self.td_token:
                if "bbox" in cells[bbox_idx] and len(cells[bbox_idx]["tokens"]) > 0:
                    bbox = cells[bbox_idx]["bbox"].copy()
                    bbox = np.array(bbox, dtype=np.float32).reshape(-1)
                    bboxes[i] = bbox
                    bbox_masks[i] = 1.0
                if self.learn_empty_box:
                    bbox_masks[i] = 1.0
                bbox_idx += 1
        data["bboxes"] = bboxes
        data["bbox_masks"] = bbox_masks
        return data

    def _merge_no_span_structure(self, structure):
        """
        This code is refer from:
        https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
        """
        new_structure = []
        i = 0
        while i < len(structure):
            token = structure[i]
            if token == "<td>":
                token = "<td></td>"
                i += 1
            new_structure.append(token)
            i += 1
        return new_structure

    def _replace_empty_cell_token(self, token_list, cells):
        """
        This fun code is refer from:
        https://github.com/JiaquanYe/TableMASTER-mmocr/blob/master/table_recognition/data_preprocess.py
        """

        bbox_idx = 0
        add_empty_bbox_token_list = []
        for token in token_list:
            if token in ["<td></td>", "<td", "<td>"]:
                if "bbox" not in cells[bbox_idx].keys():
                    content = str(cells[bbox_idx]["tokens"])
                    token = self.empty_bbox_token_dict[content]
                add_empty_bbox_token_list.append(token)
                bbox_idx += 1
            else:
                add_empty_bbox_token_list.append(token)
        return add_empty_bbox_token_list


class TableMasterLabelEncode(TableLabelEncode):
    """Convert between text-label and text-index"""

    def __init__(
        self,
        max_text_length,
        character_dict_path,
        replace_empty_cell_token=False,
        merge_no_span_structure=False,
        learn_empty_box=False,
        loc_reg_num=4,
        **kwargs
    ):
        super(TableMasterLabelEncode, self).__init__(
            max_text_length,
            character_dict_path,
            replace_empty_cell_token,
            merge_no_span_structure,
            learn_empty_box,
            loc_reg_num,
            **kwargs
        )
        self.pad_idx = self.dict[self.pad_str]
        self.unknown_idx = self.dict[self.unknown_str]

    @property
    def _max_text_len(self):
        return self.max_text_len

    def add_special_char(self, dict_character):
        self.beg_str = "<SOS>"
        self.end_str = "<EOS>"
        self.unknown_str = "<UKN>"
        self.pad_str = "<PAD>"
        dict_character = dict_character
        dict_character = dict_character + [self.unknown_str, self.beg_str, self.end_str, self.pad_str]
        return dict_character


class ResizeTableImage(object):
    def __init__(self, max_len, resize_bboxes=False, infer_mode=False, **kwargs):
        super(ResizeTableImage, self).__init__()
        self.max_len = max_len
        self.resize_bboxes = resize_bboxes
        self.infer_mode = infer_mode

    def __call__(self, data):
        img = data["image"]
        height, width = img.shape[0:2]
        ratio = self.max_len / (max(height, width) * 1.0)
        resize_h = int(height * ratio)
        resize_w = int(width * ratio)
        resize_img = cv2.resize(img, (resize_w, resize_h))
        if self.resize_bboxes and not self.infer_mode:
            data["bboxes"] = data["bboxes"] * ratio
        data["image"] = resize_img
        data["src_img"] = img
        data["shape"] = np.array([height, width, ratio, ratio])
        data["max_len"] = self.max_len
        return data


class PaddingTableImage(object):
    def __init__(self, size, **kwargs):
        super(PaddingTableImage, self).__init__()
        self.size = size

    def __call__(self, data):
        img = data["image"]
        pad_h, pad_w = self.size
        padding_img = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
        height, width = img.shape[0:2]
        padding_img[0:height, 0:width, :] = img.copy()
        data["image"] = padding_img
        shape = data["shape"].tolist()
        shape.extend([pad_h, pad_w])
        data["shape"] = np.array(shape)
        return data


class TableBoxEncode(object):
    def __init__(self, in_box_format="xyxy", out_box_format="xyxy", **kwargs):
        self.in_box_format = in_box_format
        self.out_box_format = out_box_format

    def __call__(self, data):
        img_height, img_width = data["image"].shape[:2]
        bboxes = data["bboxes"]
        if self.in_box_format != self.out_box_format:
            if self.out_box_format == "xywh":
                if self.in_box_format == "xyxyxyxy":
                    bboxes = self.xyxyxyxy2xywh(bboxes)
                elif self.in_box_format == "xyxy":
                    bboxes = self.xyxy2xywh(bboxes)

        bboxes[:, 0::2] /= img_width
        bboxes[:, 1::2] /= img_height
        data["bboxes"] = bboxes
        return data

    def xyxyxyxy2xywh(self, bboxes):
        new_bboxes = np.zeros([len(bboxes), 4])
        new_bboxes[:, 0] = bboxes[:, 0::2].min()  # x1
        new_bboxes[:, 1] = bboxes[:, 1::2].min()  # y1
        new_bboxes[:, 2] = bboxes[:, 0::2].max() - new_bboxes[:, 0]  # w
        new_bboxes[:, 3] = bboxes[:, 1::2].max() - new_bboxes[:, 1]  # h
        return new_bboxes

    def xyxy2xywh(self, bboxes):
        new_bboxes = np.empty_like(bboxes)
        new_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2  # x center
        new_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2  # y center
        new_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
        new_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
        return new_bboxes


class TableImageNorm:
    """normalize image such as substract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]

        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = (img * self.scale - self.mean) / self.std
        return data
