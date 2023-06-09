"""
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from mindspore import Tensor

__all__ = ["RecCTCLabelDecode", "RecAttnLabelDecode"]


class RecCTCLabelDecode(object):
    """Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:
        character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
            (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
        use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
        blank_at_last(bool): padding with blank index (not the space index).
            If True, a blank/padding token will be appended to the end of the dictionary, so that
            blank_index = num_chars, where num_chars is the number of character in the dictionary including space char
            if used. If False, blank token will be inserted in the beginning of the dictionary, so blank_index=0.
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
        character_dict_path=None,
        use_space_char=False,
        blank_at_last=True,
        lower=False,
    ):
        self.space_idx = None
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = [c for c in "0123456789abcdefghijklmnopqrstuvwxyz"]
            self.lower = True
            print(
                "INFO: `character_dict_path` for RecCTCLabelDecode is not given. "
                'Default dict "0123456789abcdefghijklmnopqrstuvwxyz" is applied. Only number and English letters '
                "(regardless of lower/upper case) will be recognized and evaluated."
            )
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
                print(
                    "WARNING: The dict still contains space char in dict although use_space_char is set to be False, "
                    "because the space char is coded in the dictionary file ",
                    character_dict_path,
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

        self.ignore_indices = [self.blank_idx]

        self.character = {idx: c for idx, c in enumerate(char_list)}

        self.num_classes = len(self.character)

    def decode(self, char_indices, prob=None, remove_duplicate=False):
        """
        Convert to a squence of char indices to text string
        Args:
            char_indices (np.ndarray): in shape [BS, W]
        Returns:
            text
        """

        """ convert text-index into text-label. """
        texts = []
        confs = []
        batch_size = len(char_indices)
        for batch_idx in range(batch_size):
            selection = np.ones(len(char_indices[batch_idx]), dtype=bool)
            if remove_duplicate:
                selection[1:] = char_indices[batch_idx][1:] != char_indices[batch_idx][:-1]
            for ignored_token in self.ignore_indices:
                selection &= char_indices[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in char_indices[batch_idx][selection]]
            if prob is not None:
                conf_list = prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            if self.lower:
                char_list = [x.lower() for x in char_list]

            text = "".join(char_list)

            # result_list.append((text, np.mean(conf_list).tolist()))
            texts.append(text)
            confs.append(np.mean(conf_list))
        return texts, confs

    def __call__(self, preds: Union[Tensor, np.ndarray], labels=None, **kwargs):
        """
        Args:
            preds (Union[Tensor, np.ndarray]): network prediction, class probabilities in shape [BS, W, num_classes],
                where W is the sequence length.
            labels: optional
        Return:
            texts (List[Tuple]): list of string

        """
        if isinstance(preds, tuple):
            preds = preds[-1]

        if isinstance(preds, Tensor):
            preds = preds.asnumpy()

        # preds = preds.transpose([1, 0, 2]) # [W, BS, C] -> [BS, W, C]. already did in model head.
        pred_indices = preds.argmax(axis=-1)
        pred_prob = preds.max(axis=-1)

        # print('pred indices: ', pred_indices)
        # print('pred prob: ', pred_prob.shape)

        # TODO: for debug only
        raw_chars = [[self.character[idx] for idx in pred_indices[b]] for b in range(pred_indices.shape[0])]

        texts, confs = self.decode(pred_indices, pred_prob, remove_duplicate=True)

        return {"texts": texts, "confs": confs, "raw_chars": raw_chars}


class RecAttnLabelDecode:
    def __init__(
        self, character_dict_path: Optional[str] = None, use_space_char: bool = False, lower: bool = False
    ) -> None:
        """
        Convert text label (str) to a sequence of character indices according to the char dictionary

        Args:
            character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
                (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
            use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
            lower (bool): if True, all upper-case chars in the label text will be converted to lower case.
                Set to be True if dictionary only contains lower-case chars.
                Set to be False if not and want to recognition both upper-case and lower-case.

        Attributes:
            go_idx: the index of the GO token
            stop_idx: the index of the STOP token
            num_valid_chars: the number of valid characters (including space char if used) in the dictionary
            num_classes: the number of classes (which valid characters char and the speical token for blank padding).
                so num_classes = num_valid_chars + 1
        """
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = list("0123456789abcdefghijklmnopqrstuvwxyz")

            self.lower = True
            print("INFO: The character_dict_path is None, model can only recognize number and lower letters")
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
            self.space_idx = len(char_list) + 1
        else:
            if " " in char_list:
                print(
                    "WARNING: The dict still contains space char in dict although use_space_char is set to be False, "
                    "because the space char is coded in the dictionary file ",
                    character_dict_path,
                )

        self.num_valid_chars = len(char_list)  # the number of valid chars (including space char if used)

        special_token = ["<GO>", "<STOP>"]
        char_list = special_token + char_list

        self.go_idx = 0
        self.stop_idx = 1

        self.character = {idx: c for idx, c in enumerate(char_list)}

        self.num_classes = len(self.character)

    def decode(self, char_indices: np.ndarray, probs: np.ndarray) -> Tuple[List[str], List[float]]:
        texts = list()
        confs = list()

        batch_size = len(char_indices)
        for batch_idx in range(batch_size):
            char_list = [self.character[i] for i in char_indices[batch_idx]]

            if self.lower:
                char_list = [x.lower() for x in char_list]

            text = "".join(char_list)

            pred_EOS = text.find("<STOP>")

            if pred_EOS != -1:
                text = text[:pred_EOS]
            else:
                text = ""

            if probs is not None and pred_EOS != -1:
                conf_list = probs[batch_idx][:pred_EOS]
            else:
                conf_list = [0]

            texts.append(text)
            confs.append(np.mean(conf_list))
        return texts, confs

    def __call__(self, preds: Union[Tensor, np.ndarray], labels=None, **kwargs) -> Dict[str, Any]:
        """
        Args:
            preds (dict or tuple): containing prediction tensor in shape [BS, W, num_classes]
        Return:
            texts (List[Tuple]): list of string
        """
        if isinstance(preds, tuple):
            preds = preds[-1]

        if isinstance(preds, Tensor):
            preds = preds.asnumpy()

        pred_indices = preds.argmax(axis=-1)
        pred_probs = preds.max(axis=-1)

        raw_chars = [[self.character[idx] for idx in pred_indices[b]] for b in range(pred_indices.shape[0])]

        texts, confs = self.decode(pred_indices, pred_probs)

        return {"texts": texts, "confs": confs, "raw_chars": raw_chars}


if __name__ == "__main__":
    dec = RecCTCLabelDecode()
    idx = np.array([[0, 1, 2, 10, 11, 12, 36, 36, 36, 36], [0, 1, 3, 10, 11, 12, 13, 36, 36, 36]])

    # onehot
    num_classes = np.max(idx) + 1
    preds = np.eye(num_classes)[idx]

    print(preds.shape)
    preds = preds.transpose(1, 0, 2)

    texts = dec(preds)

    print(texts)
