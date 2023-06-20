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
    """
    Format annotations into standard format for TextOCR dataset.
    The ground truths are provided as JSON files separately for training and validation,
        TextOCR_0.1_train.json, TextOCR_0.1_val.json.
    The first-level keys are "imgs", "imgToAnns", and "anns". Each image and each annotation is assigned an ID.
    The "imgs" key holds information about each image, in the form of a dictionary with the format:
        {
            "<image_id>": {"<attribute_1>": "<value_1>", ... , "<attribute_n>": "<value_n>"}, ...
        }
    The "imgToAnns" key holds the IDs of all annotations for each image in the form of a dictionary with the format:
        {
            "<image_id>": [<annotation_id_1>, ... , <annotation_id_n>], ...
        }
    The "anns" key holds information about the annotation polygon in the form of a dictionary with the format:
        {
            "<annotation_id>": {"<attribute_1>": "<value_1>", ... , "<attribute_n>": "<value_n>"}, ...
        }
    The "points" attribute of each annotation is a list of the coordinates of the annotation polygon in the format:
        `x1, y1, x2, y2, ... , xn, yn`
    The "bbox" attribute of each annotation is a list of the format:
        `x-ccord, y-coord, width, height`
    If the "points" attribute does not define a quadrilateral, the bbox attributes are used for the label.
    If the transcription is provided as ".", it is not taken into account and marked as "###".
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
