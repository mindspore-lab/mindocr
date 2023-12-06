import numpy as np

from mindspore import Tensor
from mindspore.ops import AllReduce


def load_vqa_bio_label_maps(label_map_path):
    """
    load VQA bio label maps from file.
    """
    with open(label_map_path, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
    old_lines = [line.strip() for line in lines]
    lines = ["O"]
    for line in old_lines:
        # "O" has already been in lines
        if line.upper() in ["OTHER", "OTHERS", "IGNORE"]:
            continue
        lines.append(line)
    labels = ["O"]
    for line in lines[1:]:
        labels.append("B-" + line)
        labels.append("I-" + line)
    label2id_map = {label.upper(): idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label.upper() for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


class Synchronizer:
    """
    When using distributed training, ensure all devices are synchronized.
    """

    def __init__(self, rank_size: int = 1):
        self.all_reduce = AllReduce()
        self.rank_size = rank_size

    def __call__(self):
        if self.rank_size <= 1:
            return
        else:
            sync = Tensor(np.array([1]).astype(np.int32))
            sync = self.all_reduce(sync)
            sync = sync.asnumpy()[0]
            if sync != self.rank_size:
                raise ValueError(
                    f"For Synchronizer, the sync value is not equal to rank size {self.rank_size}. "
                    f"There might be wrong with the distributed devices."
                )
