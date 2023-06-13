"""
"""
from typing import Union

import numpy as np

import mindspore as ms
from mindspore import Tensor

__all__ = ["ABINetLabelDecode"]


class ABINetLabelDecode(object):
    def __init__(
        self,
        character_dict_path=None,
        use_space_char=False,
        blank_at_last=True,
        lower=False,
    ):
        self.space_idx = None
        self.lower = lower
        self.charset = CharsetMapper(max_length=26)

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

    def decode(self, logit):
        """Greed decode"""
        # TODO: test running time and decode on GPU
        ms_softmax = ms.ops.Softmax(axis=2)
        out = ms_softmax(logit)
        pt_text, pt_scores, pt_lengths = [], [], []
        for o in out:
            text = self.charset.get_text(o.argmax(axis=1), padding=False, trim=False)
            text = text.split(self.charset.null_char)[0]  # end at end-token
            pt_text.append(text)
            pt_scores.append(o.max(axis=1)[0])
            pt_lengths.append(min(len(text) + 1, 26))  # one for end-token
        ms_stack = ms.ops.Stack()
        pt_scores = ms_stack(pt_scores)

        # pt_lengths = pt_scores.new_tensor(pt_lengths, dtype=torch.long)#我理解的就是把pt_lengths复制了一下
        pt_lengths = ms.Tensor(pt_lengths, dtype=ms.int64)

        return pt_text, pt_scores, pt_lengths

    def __call__(self, preds: Union[Tensor, np.ndarray], labels=None, **kwargs):
        """
        Args:
            preds (Union[Tensor, np.ndarray]): network prediction, class probabilities in shape [BS, W, num_classes],
                where W is the sequence length.
            labels: optional
        Return:
            texts (List[Tuple]): list of string

        """
        output = preds
        output = self._get_output(output)
        logits, pt_lengths = output[0], output[1]
        pt_text, pt_scores, pt_lengths_ = self.decode(logits)
        assert (pt_lengths == pt_lengths_).all(), f"{pt_lengths} != {pt_lengths_} for {pt_text}"

        # last_output = self._update_output(output, {'pt_text':pt_text, 'pt_scores':pt_scores})
        pt_text = [self.charset.trim(t) for t in pt_text]

        return {"texts": pt_text}

    def _get_output(self, last_output):
        if isinstance(last_output, (tuple, list)):
            for res in last_output:
                for i in range(len(res)):
                    if len(res) == 3:
                        if res[i][3] == "alignment":
                            output = res[i]
        else:
            output = last_output
        return output

    def _update_output(self, last_output, items):
        if isinstance(last_output, (tuple, list)):
            res = last_output
            if res[3] == "alignment":
                res.update(items)
        else:
            last_output.update(items)
        return last_output


class CharsetMapper(object):
    def __init__(self, max_length=30, null_char="\u2591"):
        self.null_char = null_char
        self.max_length = max_length
        self.label_to_char = self._read_charset()
        self.char_to_label = dict(map(reversed, self.label_to_char.items()))
        self.num_classes = len(self.label_to_char)

    def _read_charset(self):
        charset = {}
        charset = {
            0: "░",
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "f",
            7: "g",
            8: "h",
            9: "i",
            10: "j",
            11: "k",
            12: "l",
            13: "m",
            14: "n",
            15: "o",
            16: "p",
            17: "q",
            18: "r",
            19: "s",
            20: "t",
            21: "u",
            22: "v",
            23: "w",
            24: "x",
            25: "y",
            26: "z",
            27: "1",
            28: "2",
            29: "3",
            30: "4",
            31: "5",
            32: "6",
            33: "7",
            34: "8",
            35: "9",
            36: "0",
        }
        self.null_label = 0
        charset[self.null_label] = self.null_char
        return charset

    def trim(self, text):
        assert isinstance(text, str)
        return text.replace(self.null_char, "")

    def get_text(self, labels, length=None, padding=True, trim=False):
        """Returns a string corresponding to a sequence of character ids."""
        length = length if length else self.max_length
        labels = [int(a) if isinstance(a, ms.Tensor) else int(a) for a in labels]
        if padding:
            labels = labels + [self.null_label] * (length - len(labels))
        text = "".join([self.label_to_char[label] for label in labels])
        if trim:
            text = self.trim(text)
        return text

    def get_labels(self, text, length=None, padding=True, case_sensitive=False):
        """Returns the labels of the corresponding text."""
        length = length if length else self.max_length
        if padding:
            text = text + self.null_char * (length - len(text))
        if not case_sensitive:
            text = text.lower()
        labels = [self.char_to_label[char] for char in text]
        return labels

    def pad_labels(self, labels, length=None):
        length = length if length else self.max_length
        return labels + [self.null_label] * (length - len(labels))

    @property
    def digits(self):
        return "0123456789"

    @property
    def digit_labels(self):
        return self.get_labels(self.digits, padding=False)

    @property
    def alphabets(self):
        all_chars = list(self.char_to_label.keys())
        valid_chars = []
        for c in all_chars:
            if c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
                valid_chars.append(c)
        return "".join(valid_chars)

    @property
    def alphabet_labels(self):
        return self.get_labels(self.alphabets, padding=False)


def onehot(label, depth, device=None):
    label_shape = 26

    onehot_output = np.zeros((label_shape, depth))

    label_expand = np.expand_dims(label, -1)
    label_expand = np.expand_dims(label_expand, -1)
    label_expand_onehot = np.zeros((26, 37))
    a = 0
    for i in label_expand:
        i = int(i)
        label_expand_onehot[a][i] = 1
        a = a + 1

    label_expand_onehot = label_expand_onehot
    onehot_output = label_expand_onehot + onehot_output

    return onehot_output
