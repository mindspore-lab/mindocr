import glob
import json
import os

from tqdm import tqdm

from mindocr.data.utils.polygon_utils import sort_clockwise


class CASIA10K_Converter(object):
    """
    Format annotation to standard form for CASIA10K dataset.
    The ground truth is provided in terms of word bounding boxes. Bounding boxes are specified by the coordinates of
    their four corners in a clock-wise manner. For each image a corresponding UTF-8 encoded text file is provided,
    following the naming convention:
        `[image name].txt`
    The text files are comma separated files, where each line corresponds to one text block in the image and gives
    its bounding box coordinates (four corners, clockwise) and its transcription in the format:
        `x1,y1,x2,y2,x3,y3,x4,y4,transcription`
    Note that the transcription is anything that follows the 8th comma until the end of line. No escape characters are
    to be used.
    If the transcription is provided as "###", then text block (word) is considered as "don't care".
    This dataset is a Chinese text dataset.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        else:
            raise ValueError("CASIA10K currently only support detection.")

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        with open(output_path, "w") as out_file:
            for label_fp in tqdm(label_paths):
                label_file_name = os.path.basename(label_fp)
                img_path = os.path.join(image_dir, label_file_name.split(".")[0] + ".jpg")
                assert os.path.exists(
                    img_path
                ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                label = []
                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)
                with open(label_fp, "r", encoding="gbk", errors="ignore") as f:
                    for line in f.readlines():
                        tmp = line.strip("\n\r").split(",", 8)

                        points = [[int(tmp[i]), int(tmp[i + 1])] for i in range(0, 8, 2)]
                        points = sort_clockwise(points).tolist()

                        label.append({"transcription": tmp[8], "points": points})

                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
