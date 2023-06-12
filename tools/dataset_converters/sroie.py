import glob
import json
import os

from shapely.geometry import Polygon


class SROIE_Converter(object):
    """
    Format annotation to standard form for SROIE dataset.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        else:
            raise ValueError("SROIE currently only support detection.")

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))

        processed = 0
        with open(output_path, "w") as out_file:
            for label_fp in label_paths:
                label_file_name = os.path.basename(label_fp)
                img_path = os.path.join(image_dir, label_file_name.split(".")[0] + ".jpg")

                if not os.path.exists(img_path):
                    print(f"Warning: {os.path.basename(img_path)} not found")
                    continue

                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)

                label = []
                with open(label_fp, "r", encoding="gbk") as f:
                    for line in f.readlines():
                        tmp = line.strip("\n\r").split(",", 8)
                        if len(tmp) == 1:   # skip empty lines
                            continue

                        points = [[int(tmp[i]), int(tmp[i + 1])] for i in range(0, 8, 2)]
                        if not Polygon(points).is_valid:
                            print(f"Warning {os.path.basename(img_path)}: skipping invalid polygon {points}")
                            continue

                        label.append({"transcription": tmp[8] if tmp[8] != "***" else "###", "points": points})

                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")
                processed += 1

        print(f"Processed {processed} images.")
