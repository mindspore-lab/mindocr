import os
import sys
import warnings
from typing import Sequence

import numpy as np

# add mindocr root path, and import postprocess from mindocr
mindocr_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
sys.path.insert(0, mindocr_path)

from mindocr.postprocess import rec_postprocess  # noqa

# adapted to ppocr/mmocr postprocess
__all__ = ["RecCTCLabelDecode", "ViTSTRLabelDecode", "AttentionLabelDecode"]


class RecCTCLabelDecode(rec_postprocess.RecCTCLabelDecode):
    """RecCTCLabelDecode, adapted to paddleocr"""

    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        blank_at_last=True,
        lower=False,
        # for paddleocr
        use_redundant_space_char=False,
    ):
        super().__init__(
            character_dict_path=character_dict_path,
            use_space_char=use_space_char,
            blank_at_last=blank_at_last,
            lower=lower,
        )
        if not use_redundant_space_char:
            return

        # some paddleOCR rec models need an extra space char
        self.num_valid_chars += 1

        char_list = [v for v in self.character.values()]
        if blank_at_last:
            char_list.insert(-1, " ")  # extra space in front of blank
            self.blank_idx += 1
            self.ignore_indices = [self.blank_idx]
        else:
            char_list.append(" ")

        self.character = {idx: c for idx, c in enumerate(char_list)}
        self.num_classes = len(self.character)

    def __call__(self, preds):
        """
        Args:
            preds (np.ndarray): containing prediction tensor in shape [BS, W, num_classes]
        Return:
            texts (List[Tuple]): list of string
        """
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds

        if len(preds.shape) == 3:
            pred_indices = preds.argmax(axis=-1)
        else:
            pred_indices = preds

        texts, confs = self.decode(pred_indices, remove_duplicate=True)

        return {"texts": texts, "confs": confs}


class ViTSTRLabelDecode(rec_postprocess.RecCTCLabelDecode):
    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        blank_at_last=True,
        **kwargs,
    ):
        super().__init__(character_dict_path, use_space_char, blank_at_last, **kwargs)
        char_list = list(self.character.values())[1:]
        char_list = ["<s>", "</s>"] + char_list
        self.character = {idx: c for idx, c in enumerate(char_list)}
        self.num_classes = len(self.character)

    def decode(self, char_indices, prob=None, remove_duplicate=False):
        texts = []
        confs = []
        batch_size = len(char_indices)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(char_indices[batch_idx])):
                try:
                    char_idx = self.character[int(char_indices[batch_idx][idx])]
                except Exception:
                    continue
                if char_idx == "</s>":
                    break
                char_list.append(char_idx)
                conf_list.append(prob[batch_idx][idx]) if prob is not None else conf_list.append(1)
            texts.append("".join(char_list).lower())
            confs.append(np.mean(conf_list))
        return texts, confs

    def __call__(self, preds):
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds
        preds = preds[:, 1:]
        pred_indices = preds.argmax(axis=-1)
        preds_prob = preds.max(axis=2)
        texts, confs = self.decode(pred_indices, preds_prob, remove_duplicate=True)
        return {"texts": texts, "confs": confs}


class AttentionLabelDecode(object):
    """Convert between text-label and text-index"""

    def __init__(self, character_dict_path=None, ignore_chars: Sequence[str] = ["padding"], **kwargs):
        self.character_str = []

        if character_dict_path is None:
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<="
            dict_character = list(self.character_str)
        else:
            with open(character_dict_path, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str.append(line)
            dict_character = list(self.character_str)
        self.end_idx = len(dict_character)
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character
        mapping_table = {
            "padding": self.dict["<PAD>"],
            "end": len(self.character_str),
            "unknown": self.dict["<UKN>"],
        }
        ignore_indexes = list()
        for ignore_char in ignore_chars:
            index = mapping_table.get(ignore_char)
            if index is None or (index == mapping_table.get("unknown") and ignore_char != "unknown"):
                warnings.warn(f"{ignore_char} does not exist in the dictionary", UserWarning)
                continue
            ignore_indexes.append(index)
        self.ignore_indexes = ignore_indexes

    def __call__(self, preds):
        preds = preds[0] if isinstance(preds, (tuple, list)) else preds
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        texts, confs = self.decode(preds_idx, preds_prob)
        return {"texts": texts, "confs": confs}

    def add_special_char(self, dict_character):
        dict_character = dict_character + ["<BOS/EOS>", "<PAD>", "<UKN>"]
        return dict_character

    def decode(self, text_index, text_prob=None):
        """convert text-index into text-label."""
        texts = []
        confs = []
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                try:
                    char_idx = int(text_index[batch_idx][idx])
                    char = self.character[char_idx]
                except Exception:
                    continue
                if char_idx in self.ignore_indexes:
                    continue
                if char_idx == self.end_idx:
                    break
                char_list.append(char)
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            texts.append("".join(char_list))
            confs.append(np.mean(conf_list))
        return texts, confs
