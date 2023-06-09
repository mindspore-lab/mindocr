import glob
import json
import math
import os


def rotate_xy(x, y, center_x, center_y, theta):
    rotate_x = math.cos(theta) * (x - center_x) - math.sin(theta) * (y - center_y)
    rotate_y = math.cos(theta) * (y - center_y) + math.sin(theta) * (x - center_x)
    return center_x + rotate_x, center_y + rotate_y


def det_rotate(x, y, width, height, theta):
    center_x = x + width / 2
    center_y = y + height / 2

    x1, y1 = rotate_xy(x, y, center_x, center_y, theta)
    x2, y2 = rotate_xy(x + width, y, center_x, center_y, theta)
    x3, y3 = rotate_xy(x + width, y + height, center_x, center_y, theta)
    x4, y4 = rotate_xy(x, y + height, center_x, center_y, theta)
    return x1, y1, x2, y2, x3, y3, x4, y4


class TD500_Converter(object):
    """
    Format annotation to standard form for MSRA-TD500 dataset.
    """

    def __init__(self, path_mode="relative"):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        if task == "rec":
            raise ValueError("SynText dataset has no cropped word images and recognition labels.")

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.gt")))
        with open(output_path, "w") as out_file:
            for label_fp in label_paths:
                label_file_name = os.path.basename(label_fp)
                img_path = os.path.join(image_dir, label_file_name[:-3] + ".JPG")
                assert os.path.exists(
                    img_path
                ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                label = []
                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)
                with open(label_fp, "r", encoding="utf-8-sig") as f:
                    for line in f.readlines():
                        tmp = line.strip("\n").replace("\xef\xbb\xbf", "").split(" ")
                        x1, y1, x2, y2, x3, y3, x4, y4 = det_rotate(
                            int(tmp[2]), int(tmp[3]), int(tmp[4]), int(tmp[5]), float(tmp[6])
                        )
                        s = [[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]]
                        if tmp[1] == "1":
                            result = {"transcription": "###", "points": s}
                        else:
                            result = {"transcription": tmp[1], "points": s}
                        label.append(result)

                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
