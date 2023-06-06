"""
TextOCR
https://textvqa.org/textocr/
"""
import json
from pathlib import Path

from shapely.geometry import Polygon
from tqdm import tqdm

from mindocr.data.utils.polygon_utils import sort_clockwise


class TEXTOCR_Converter:
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

        with open(output_path, "w", encoding="utf-8") as out_file:
            for image_info in tqdm(data["imgs"].values(), total=len(data["imgs"])):
                img_path = image_dir / Path(image_info["file_name"]).name
                assert img_path.exists(), f"Image {img_path} not found."

                label = []
                annotations = data["imgToAnns"][image_info["id"]]
                for anno_id in annotations:
                    anno = data["anns"][anno_id]
                    points = [
                        [int(anno["points"][i]), int(anno["points"][i + 1])] for i in range(0, len(anno["points"]), 2)
                    ]  # reshape points (N, 2)

                    poly = Polygon(points)
                    if not poly.is_valid:  # fix broken polygons
                        if len(points) == 4:  # if it's a quadrilateral - fix the polygon
                            points = sort_clockwise(points).tolist()
                        else:  # else take the bounding box as the label
                            x, y, w, h = anno["bbox"]
                            points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

                    elif not poly.exterior.is_ccw:  # sort vertices in polygons in clockwise order
                        points = points[::-1]

                    # a single dot sign is an ignore tag in TextOCR
                    label.append(
                        {
                            "transcription": anno["utf8_string"] if anno["utf8_string"] != "." else "###",
                            "points": points,
                        }
                    )

                img_path = img_path.name if self._relative else str(img_path)
                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
