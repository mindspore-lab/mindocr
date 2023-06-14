import numpy as np


class RecCTCLabelDecode(object):
    """Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:
        character_dict_path: path to dictionary, if None, a dictionary containing 36 chars
            (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
        use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
        blank_at_last(bool): padding with blank index (not the space index). If True, a blank/padding token will be
            appended to the end of the dictionary, so that blank_index = num_chars, where num_chars is the number of
            character in the dictionary including space char if used. If False, blank token will be inserted in the
            beginning of the dictionary, so blank_index=0.
        lower (bool): if True, all upper-case chars in the label text will be converted to lower case. Set to be True
            if dictionary only contains lower-case chars. Set to be False if not and want to recognition both upper-case
            and lower-case.

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

    def decode(self, char_indices, remove_duplicate=False):
        """
        Convert to a squence of char indices to text string
        Args:
            char_indices (np.ndarray): in shape [BS, W]
        Returns:
            text
        """

        """ convert text-index into text-label. """
        texts = []
        batch_size = len(char_indices)
        for batch_idx in range(batch_size):
            selection = np.ones(len(char_indices[batch_idx]), dtype=bool)
            if remove_duplicate:
                selection[1:] = char_indices[batch_idx][1:] != char_indices[batch_idx][:-1]
            for ignored_token in self.ignore_indices:
                selection &= char_indices[batch_idx] != ignored_token

            char_list = [self.character[text_id] for text_id in char_indices[batch_idx][selection]]

            text = "".join(char_list)

            # result_list.append((text, np.mean(conf_list).tolist()))
            texts.append(text)

        return texts

    def __call__(self, preds: np.ndarray):
        """
        Args:
            preds (dict or tuple): containing prediction tensor in shape [W, BS, num_classes]
            labels ():
        Return:
            texts (List[Tuple]): list of string

        """
        if len(preds.shape) == 3:
            pred_indices = preds.argmax(axis=-1)
        else:
            pred_indices = preds

        texts = self.decode(pred_indices, remove_duplicate=True)

        return texts
