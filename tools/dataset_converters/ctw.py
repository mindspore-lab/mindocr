"""
A Large Chinese Text Dataset in the Wild
https://ctwdataset.github.io/
"""
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


class CTW_Converter:
    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        if task == "rec":
            raise ValueError("Not implemented")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            with open(label_path, "r") as json_file:
                for line in tqdm(json_file):
                    line = json.loads(line)

                    img_path = image_dir / line["file_name"]
                    assert img_path.exists(), f"Image {img_path} not found."

                    label = []
                    for annotation in line["annotations"]:
                        sentence = "".join([a["text"] for a in annotation])
                        char_polys = np.array([a["polygon"] for a in annotation], dtype=np.float32).reshape(-1, 2)
                        # convert character polygons to a sentence polygon
                        hull = cv2.convexHull(char_polys).squeeze(1)

                        label.append({"transcription": sentence, "points": hull.tolist()})

                    for annotation in line["ignore"]:
                        label.append({"transcription": "###", "points": annotation["polygon"]})

                    img_path = img_path.name if self._relative else str(img_path)
                    out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
