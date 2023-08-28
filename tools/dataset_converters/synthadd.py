import os
import re

from tools.dataset_converters.utils.lmdb_writer import create_lmdb_dataset


class SYNTHADD_Converter:
    def __init__(self, *args, **kwargs):
        self._image_dir = None

    def convert(self, task="rec_lmdb", image_dir=None, label_path=None, output_path=None):
        if task == "rec_lmdb":
            self.convert_rec_lmdb(image_dir, output_path)
        else:
            raise ValueError(f"Unsupported task `{task}`.")

    def convert_rec_lmdb(self, image_dir=None, output_path=None):
        self._image_dir = image_dir

        folders = [f"crop_img_{i}" for i in range(1, 21)]
        annotations = [f"annotationlist/gt_{i}.txt" for i in range(1, 21)]
        folders = [os.path.join(image_dir, x) for x in folders]
        annotations = [os.path.join(image_dir, x) for x in annotations]

        images, labels = [], []
        for folder, anno in zip(folders, annotations):
            with open(anno, "r") as f:
                for line in f:
                    content = re.findall(r"(\w+.jpg),\"(.+)\"\n", line)
                    assert len(content) == 1, line
                    image_path, label = content[0]
                    image_path = os.path.join(folder, image_path)
                    images.append(image_path)
                    labels.append(label)

        create_lmdb_dataset(images, labels, output_path)
