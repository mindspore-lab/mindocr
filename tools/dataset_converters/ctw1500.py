import glob
import json
import os
import xml.etree.ElementTree as ET


class CTW1500_Converter(object):
    """
    Format annotation to standard form for CTW-1500 dataset.
    """

    def __init__(self, path_mode="relative"):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        else:
            raise ValueError("ctw1500 currently only support detection.")

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        if label_paths:
            with open(output_path, "w") as out_file:
                for label_fp in label_paths:
                    label_file_name = os.path.basename(label_fp)
                    img_path = os.path.join(image_dir, label_file_name.split(".")[0][3:] + ".jpg")
                    assert os.path.exists(
                        img_path
                    ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                    label = []
                    if self.path_mode == "relative":
                        img_path = os.path.basename(img_path)
                    with open(label_fp, "r", encoding="utf-8-sig") as f:
                        for line in f.readlines():
                            tmp = line.strip("\n\r").split(",####")
                            assert len(tmp), f"parse error for {tmp}."
                            points = tmp[0].split(",")
                            assert (
                                len(points) % 2 == 0
                            ), f"The length of the points should be an even number, but get {len(points)}"
                            s = []
                            for i in range(0, len(points), 2):
                                b = [int(points[i]), int(points[i + 1])]
                                s.append(b)
                            result = {"transcription": tmp[-1], "points": s}
                            label.append(result)

                    out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")

        else:
            label_paths = sorted(glob.glob(os.path.join(label_dir, "*.xml")))
            with open(output_path, "w") as out_file:
                for label_fp in label_paths:
                    label_file_name = os.path.basename(label_fp)
                    img_path = os.path.join(image_dir, label_file_name.split(".")[0] + ".jpg")
                    assert os.path.exists(
                        img_path
                    ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                    label = []
                    if self.path_mode == "relative":
                        img_path = os.path.basename(img_path)
                    tree = ET.parse(label_fp)
                    for obj in tree.findall("image"):
                        for tmp in obj.findall("box"):
                            annotation = tmp.find("label").text
                            points = tmp.find("segs").text.split(",")

                            assert len(points) == 28, f"The length of the points should be 28, but get {len(points)}"
                            s = []
                            for i in range(0, len(points), 2):
                                b = [int(points[i]), int(points[i + 1])]
                                s.append(b)
                            result = {"transcription": annotation, "points": s}
                            label.append(result)

                    out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
