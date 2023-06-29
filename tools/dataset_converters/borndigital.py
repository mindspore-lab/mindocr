"""
BDI - Born-Digital Images
https://rrc.cvc.uab.es/?ch=1
"""
import json
from pathlib import Path

import cv2
from shapely.geometry import Polygon
from tqdm import tqdm


class BORNDIGITAL_Converter:
    """
    Format annotation to standard form for Born-Digital dataset.
    The ground truth is provided in terms of word bounding boxes. Bounding boxes are specified by the coordinates of
    their two opposite corners. For each image a corresponding UTF-8 encoded text file is provided,
    following the naming convention:
        `gt_[image name].txt`
    The text files are comma separated files, where each line corresponds to one text block in the image and gives
    its bounding box coordinates (top-right corner and bottom-left corner) and its transcription in the format:
        `x1,y1,x2,y2,transcription`
    Language is English only.
    Note that the dataset also contains .gif files which are also handled by the converter.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        else:
            raise ValueError("Born-Digital currently supports only detection.")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem.split("_")[-1]))
            for img_path in tqdm(images, total=len(images)):
                label = []
                with open(label_path / ("gt_" + img_path.stem + ".txt"), "r", encoding="utf-8") as label_file:
                    for line in label_file.readlines():
                        line = line.split(", ", 4)  # Split the line into 5 parts (4 points + transcription)
                        line[:4] = [int(x) for x in line[:4]]

                        points = [[line[0], line[1]], [line[2], line[1]], [line[2], line[3]], [line[0], line[3]]]
                        if not Polygon(points).is_valid:
                            print(f"Warning {img_path.name}: skipping invalid polygon {line[:4]}")
                            continue

                        label.append(
                            {"transcription": line[4].strip()[1:-1], "points": points}
                        )  # strip newline character and the default quotation marks

                # Handle gif animation by storing as new image via video capturing
                if img_path.suffix.lower() == ".gif":
                    new_path = image_dir / (img_path.stem + ".png")
                    if not new_path.exists():
                        cap = cv2.VideoCapture(str(img_path))
                        _, image = cap.read()
                        cap.release()

                        cv2.imwrite(str(new_path), image)
                        img_path = new_path
                    else:
                        continue

                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
