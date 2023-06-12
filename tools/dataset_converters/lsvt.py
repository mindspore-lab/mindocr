import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon


class LSVT_Converter:
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
        with open(label_path, "r") as json_file:
            data = json.load(json_file)

        processed = 0
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem.split("_")[-1]))  # sort by image id
            for img_path in images:
                image_info = data[img_path.stem]

                label = []
                for item in image_info:
                    if len(item["points"]) < 3:
                        print(f"Warning {img_path.name}: skipping invalid polygon {item['points']}")
                        continue

                    if not Polygon(item["points"]).is_valid:
                        # TODO: a better way to fix invalid polygons?
                        print(f"Warning {img_path.name}: invalid polygon. Fixing it with convex hull.")
                        item["points"] = cv2.convexHull(np.array(item["points"])).squeeze(1).tolist()

                    label.append(
                        {
                            "transcription": item["transcription"] if not item["illegibility"] else "###",
                            "points": item["points"],
                        }
                    )

                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
                processed += 1

        print(f"Processed {processed} images.")
