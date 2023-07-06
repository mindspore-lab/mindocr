"""
ReCTS - ICDAR 2019 Robust Reading Challenge on Reading Chinese Text on Signboard
https://rrc.cvc.uab.es/?ch=12
"""
import json
from pathlib import Path

from shapely.geometry import Polygon


class RECTS_Converter:
    """
    Format annotations into the standard form for the RECTS dataset.
    The ground truth is provided as JSON files corresponding to each image with naming format:
        `<image name>.json`
    Each JSON file has the following structure:
        ``` {
            chars: List[dict{<anno_attrib, value>}]
            lines: List[dict{<anno_attrib, value>}]
        } ```
    Note that both the character-level and line-level annotations are provided in the dataset. This converter
    uses the line-level annotations.
    Each dict{} in the lines list corresponds to a single line text instance in the image.
    The 'points' attribute of each dict{} gives the bounding box coordinates for each text instance in a clockwise
    manner with the following format:
        `[x1, y1, x2, y2, x3, y3, x4, y4]`
    If the 'ignore' attribute is provided as 1, the text instance is considered as "###".
    """

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
        processed = 0
        with open(output_path, "w", encoding="utf-8") as out_file:
            images = sorted(image_dir.iterdir(), key=lambda path: int(path.stem.split("_")[-1]))  # sort by image id
            for img_path in images:
                with open(label_path / (img_path.stem + ".json"), "r") as f:
                    image_info = json.load(f)

                label = []
                for line in image_info["lines"]:
                    points = [
                        [int(line["points"][i]), int(line["points"][i + 1])] for i in range(0, len(line["points"]), 2)
                    ]  # reshape points (4, 2)

                    if not Polygon(points).is_valid:
                        print(f"Warning {img_path.name}: skipping invalid polygon {points}")
                        continue

                    label.append(
                        {
                            "transcription": line["transcription"] if not line["ignore"] else "###",
                            "points": points,
                        }
                    )

                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
                processed += 1

        print(f"Processed {processed} images.")
