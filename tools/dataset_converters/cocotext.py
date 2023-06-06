"""
COCO-Text - ICDAR2017 Robust Reading Challenge on COCO-Text
https://rrc.cvc.uab.es/?ch=5
"""
import json
from pathlib import Path

from shapely.geometry import Polygon

from mindocr.data.utils.polygon_utils import sort_clockwise


class COCOTEXT_Converter:
    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

        if "split" not in kwargs:
            raise ValueError("It is required to specify the `split` argument for the COCO-Text dataset converter.")
        self._split = kwargs["split"]

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
            for image_info in data["imgs"].values():
                if image_info["set"] != self._split:
                    continue

                annotations = data["imgToAnns"][str(image_info["id"])]
                if not annotations:  # skip images without text instances
                    continue

                img_path = image_dir / image_info["file_name"]
                assert img_path.exists(), f"Image {img_path} not found."

                label = []
                all_illegible = True  # ensure that there's at least one legible text instance in an image
                for anno_id in annotations:
                    anno = data["anns"][str(anno_id)]
                    points = [
                        [int(anno["polygon"][i]), int(anno["polygon"][i + 1])]
                        for i in range(0, len(anno["polygon"]), 2)
                    ]  # reshape points (N, 2)

                    points = sort_clockwise(points).tolist()  # fix broken polygons
                    if not Polygon(points).is_valid:
                        print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                        continue

                    if "utf8_string" not in anno:  # if a text instance is not readable
                        anno["utf8_string"] = "###"

                    label.append(
                        {
                            "type": anno["class"],  # machine printed / handwritten / others
                            "language": anno["language"],  # English / non-English / N/A
                            "transcription": anno["utf8_string"] if anno["legibility"] == "legible" else "###",
                            "points": points,
                        }
                    )

                    all_illegible = all_illegible and (label[-1]["transcription"] == "###")

                if not all_illegible:
                    processed += 1
                    img_path = img_path.name if self._relative else str(img_path)
                    out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")

        print(f"Processed {processed} images.")
