import json
from pathlib import Path

import cv2
from shapely.geometry import Polygon
from tqdm import tqdm

from mindocr.data.utils.polygon_utils import sort_clockwise


class MLT2017_Converter:
    """
    Format annotation to standard form for MLT2017 dataset.
    The ground truth is provided in terms of word bounding boxes. Bounding boxes are specified by the coordinates of
    their four corners in a clock-wise manner. For each image a corresponding UTF-8 encoded text file is provided,
    following the naming convention:
        `gt_[image name].txt`
    The text files are comma separated files, where each line corresponds to one text block in the image and gives
    its bounding box coordinates (four corners, clockwise), its script and its transcription in the format:
        `x1,y1,x2,y2,x3,y3,x4,y4,script,transcription`
    Valid scripts are: "Arabic", "Latin", "Chinese", "Japanese", "Korean", "Bangla", "Symbols", "Mixed", "None"
    Note that the transcription is anything that follows the 9th comma until the end of line. No escape characters are
    to be used.
    If the transcription is provided as "###", then text block (word) is considered as "don't care".
    """

    def __init__(self, path_mode="relative"):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        if task == "rec":
            self._format_rec_label(label_path, output_path)

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem.split("_")[-1]))  # sort by image id
            for img_path in tqdm(images, total=len(images)):
                label = []
                with open(label_path / ("gt_" + img_path.stem + ".txt"), "r", encoding="utf-8") as f:
                    for line in f.read().splitlines():
                        line = line.split(",", 9)  # split the line by first 9 commas: 8 points + language

                        points = [[int(line[i]), int(line[i + 1])] for i in range(0, 8, 2)]  # reshape points (4, 2)
                        # sort points and validate
                        points = sort_clockwise(points).tolist()
                        if not Polygon(points).is_valid:
                            print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                            continue

                        label.append({"language": line[8], "transcription": line[9], "points": points})

                # gif is animation not an image, save it as an image
                if img_path.suffix.lower() == ".gif":
                    new_path = image_dir / (img_path.stem + ".png")
                    if not new_path.exists():  # if was not converted previously
                        cap = cv2.VideoCapture(str(img_path))
                        _, image = cap.read()
                        cap.release()

                        cv2.imwrite(str(new_path), image)
                        img_path = new_path
                    else:  # skip .gif image since converted .png already exists in the folder
                        continue

                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")

    def _format_rec_label(self, label_path, output_path):
        with open(output_path, "w") as outf:
            with open(label_path, "r") as f:
                for line in f:
                    # , may occur in text
                    sep_index = line.find(",")
                    img_path = line[:sep_index].strip().replace("\ufeff", "")
                    label = line[sep_index + 1 :].strip().replace('"', "")
                    sep_index = label.find(",")
                    language = label[:sep_index].strip().replace('"', "")
                    label = label[sep_index + 1 :].strip().replace('"', "")

                    outf.write(img_path + "\t" + language + "\t" + label + "\n")
