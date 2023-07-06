import json
from pathlib import Path

from tqdm import tqdm


class CCPD_Converter:
    """
    Format annotation to standard form for CCPD2019 dataset.
    CCPD2019 is a dataset for license plate (lp) text detection and recognition.
    The ground truths are embedded into the filenames of the images of the dataset, so there are no additional
    annotation file(s). Thus, the filenames follow a specific format:

        `<area ratio>-<tilt>-<bbox coords>-<vertices>-<lp number>-<brightness>-<blurriness>`

    The 'area ratio', 'brightness', and 'blurriness' properties are simple integers.
    The 'tilt' property is split into further two: `<horizontal tilt>_<vertical tilt>`.
    The 'bbox coords' property provides top-left and bottom-right coords, respectively: `<x1>&<y1>_<x2>&<y2>`
    The 'vertices' property provides the points for the polygon: `<x1>&<y1>_<x2>&<y2>_<x3>&<y3>_<x4>&<y4>`
    The 'lp number' property provides the transcription as explained here:
        https://github.com/detectRecog/CCPD#dataset-annotations

    Each image is assumed to have only one license plate (lp). The information about the one lp is embedded
    into the file name of the image.
    Note: the lp number consists of a province as a Chinese character and the remaining characters are English
    alphanumeric characters.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"
        self.provinces = [
            "皖",
            "沪",
            "津",
            "渝",
            "冀",
            "晋",
            "蒙",
            "辽",
            "吉",
            "黑",
            "苏",
            "浙",
            "京",
            "闽",
            "赣",
            "鲁",
            "豫",
            "鄂",
            "湘",
            "粤",
            "桂",
            "琼",
            "川",
            "贵",
            "云",
            "藏",
            "陕",
            "甘",
            "青",
            "宁",
            "新",
            "警",
            "学",
            "O",
        ]
        self.alphabets = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "O",
        ]
        self.ads = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "J",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "O",
        ]

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "det":
            self._format_det_label(Path(image_dir), label_path, output_path)
        else:
            raise ValueError("The CCPD dataset currently supports detection only!")

    def _format_det_label(self, image_dir: Path, label_path: Path, output_path: str):
        with open(output_path, "w", encoding="utf-8") as out_file:
            with open(label_path, "r") as f:
                for line in tqdm(f.readlines()):
                    img_path = image_dir / line.strip()
                    assert img_path.exists(), f"Image {img_path} not found."

                    area, tilt, bbox, vertices, lp, brightness, blurriness = img_path.stem.split("-")
                    h_tilt, v_tilt = [int(x) for x in tilt.split("_")]
                    bbox = [
                        [int(x) for x in coordinates.split("&")] for coordinates in bbox.split("_")
                    ]  # reshape (2, 2)

                    points = [
                        [int(x) for x in coordinates.split("&")] for coordinates in vertices.split("_")
                    ]  # reshape (N, 2)

                    province = self.provinces[int(lp.split("_")[0])]
                    alphabet = self.alphabets[int(lp.split("_")[1])]
                    ad = ""
                    for i in lp.split("_")[2:]:
                        ad += self.ads[int(i)]
                    lp_text = province + alphabet + ad

                    label = {
                        "area": int(area),
                        "h_tilt": h_tilt,
                        "v_tilt": v_tilt,
                        "bbox": bbox,
                        "points": points,
                        "transcription": lp_text,
                        "brightness": int(brightness),
                        "blurriness": int(blurriness),
                    }

                    img_path = img_path.name if self._relative else str(img_path)
                    out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
