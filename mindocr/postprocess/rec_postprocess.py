"""
"""
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from mindspore import Tensor

__all__ = ["RecCTCLabelDecode", "RecAttnLabelDecode", "RecMasterLabelDecode", "VisionLANPostProcess"]
_logger = logging.getLogger(__name__)


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
            _logger.info(
                "`character_dict_path` for RecCTCLabelDecode is not given. "
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

        # TODO: for debug only
        raw_chars = [[self.character[idx] for idx in pred_indices[b]] for b in range(pred_indices.shape[0])]

        texts, confs = self.decode(pred_indices, pred_prob, remove_duplicate=True)

        return {"texts": texts, "confs": confs, "raw_chars": raw_chars}


class VisionLANPostProcess(RecCTCLabelDecode):
    """Convert the predicted tensor to text strings, confidence(probabilities), and raw characters
    Args:
        character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
            (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
        use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
        lower (bool): if True, all upper-case chars in the label text will be converted to lower case.
            Set to be True if dictionary only contains lower-case chars.
            Set to be False if not and want to recognition both upper-case and lower-case.
        blank_at_last(bool): padding with blank index (not the space index).
            If True, a blank/padding token will be appended to the end of the dictionary, so that
            blank_index = num_chars, where num_chars is the number of character in the dictionary including space char
            if used. If False, blank token will be inserted in the beginning of the dictionary, so blank_index=0.
        max_text_length(int): the maximum length of the text string. Default is 25.
    Attributes:
        character (dict): the dictionary of valid characters.
        max_text_length (int): the maximum length of the text string.
        num_classes (int): the number of classes (which valid characters char and the speical token for blank padding).
            so num_classes = num_valid_chars + 1
    """

    def __init__(
        self, character_dict_path=None, use_space_char=False, blank_at_last=True, lower=False, max_text_length=25
    ):
        super(VisionLANPostProcess, self).__init__(character_dict_path, use_space_char, blank_at_last, lower)
        self.max_text_length = max_text_length
        assert (
            not blank_at_last
        ), "VisionLAN uses blank_at_last =  False, please check your configuration for VisionLANPostProcess"

    def __call__(self, preds, *args, **kwargs):
        if isinstance(preds, Tensor):  # eval mode
            text_pre = preds.numpy()  # (max_len, b, 37)) before the softmax function
            b = text_pre.shape[1]
            lenText = self.max_text_length
            nsteps = self.max_text_length
            out_res = np.zeros(shape=[lenText, b, self.num_classes])
            out_length = np.zeros(shape=[b])
            now_step = 0
            for _ in range(nsteps):
                if 0 in out_length and now_step < nsteps:
                    tmp_result = text_pre[now_step, :, :]  # (b, 37)
                    out_res[now_step] = tmp_result
                    tmp_result = (-tmp_result).argsort(axis=1)[
                        :, 0
                    ]  # top1 result index at axis=1, 37 is the dictionary size
                    for j in range(b):
                        if out_length[j] == 0 and tmp_result[j] == 0:
                            # for the jth sample, if the character with greatest probibility is <PAD>,
                            # assign the now_step+1 to  out_length[j] as the prediction length
                            out_length[j] = now_step + 1
                    now_step += 1
            for j in range(0, b):
                if int(out_length[j]) == 0:
                    out_length[j] = nsteps
            start = 0
            output = np.zeros((int(out_length.sum()), self.num_classes))
            for i in range(0, b):
                cur_length = int(out_length[i])
                output[start : start + cur_length] = out_res[0:cur_length, i, :]
                start += cur_length
            net_out = output
            length = out_length
        else:  # do not call postprocess in train mode
            raise ValueError("Do not call postprocess in train mode")
        texts = []
        raw_chars = []
        confs = []
        # 1. apply softmax function to net_out
        net_out = np.exp(net_out) / (np.expand_dims(np.exp(net_out).sum(1), axis=1) + 1e-7)  # (N, 37)
        for i in range(0, length.shape[0]):
            start = int(length[:i].sum())
            end = int(length[:i].sum() + length[i])
            preds_idx_r = (-net_out[start:end]).argsort(1)[:, 0]  # top1 result index at axis=1
            preds_idx = preds_idx_r.tolist()
            pred_chars = [self.character[idx] if idx > 0 and idx <= len(self.character) else "" for idx in preds_idx]
            preds_text = "".join(pred_chars)
            preds_prob = net_out[start:end].max(axis=1)  # top1 result at axis=1
            preds_prob = np.exp(np.log(preds_prob).sum() / (preds_prob.shape[0] + 1e-6))
            texts.append(preds_text)
            raw_chars.append(pred_chars)
            confs.append(preds_prob)
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

        self.character = {idx: c for idx, c in enumerate(char_list)}

        self.num_classes = len(self.character)

    def decode(self, char_indices: np.ndarray, probs: np.ndarray) -> Tuple[List[str], List[float]]:
        texts = list()
        confs = list()

        batch_size = len(char_indices)
        for batch_idx in range(batch_size):
            char_list = [self.character[i] for i in char_indices[batch_idx]]

            try:
                pred_EOS = char_list.index("<STOP>")
            except ValueError:
                pred_EOS = -1

            if self.lower:
                char_list = [x.lower() for x in char_list]

            if pred_EOS != -1:
                char_list = char_list[:pred_EOS]
                text = "".join(char_list)
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


class RecMasterLabelDecode(RecAttnLabelDecode):
    def __init__(
        self,
        character_dict_path: Optional[str] = None,
        use_space_char: bool = False,
        use_unknown_char: bool = False,
        lower: bool = False,
    ) -> None:
        """
        Convert text label (str) to a sequence of character indices according to the char dictionary

        Args:
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

        # use unknow symbol if set
        if use_unknown_char:
            char_list = char_list + [self.unknown_token]
            self.unknown_idx = len(char_list) - 1

        self.character = {idx: c for idx, c in enumerate(char_list)}

        self.num_classes = len(self.character)
