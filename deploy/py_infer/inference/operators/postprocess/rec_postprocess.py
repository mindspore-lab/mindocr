import numpy as np

from ...utils import file_base_check


class RecCTCLabelDecode(object):
    def __init__(self, character_dict_path):
        self.labels = [' ']
        file_base_check(character_dict_path)
        with open(character_dict_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip("\n").strip("\r\n")
                self.labels.append(line)
        self.labels.append(' ')

    def __call__(self, preds: np.ndarray):
        batchsize, length = preds.shape
        texts = []
        for index in range(batchsize):
            char_list = []
            for i in range(length):
                if preds[index, i] and i and preds[index, i - 1] != preds[index, i]:
                    char_list.append(self.labels[preds[index, i]])
            text = ''.join(char_list)
            texts.append(text)
        return texts
