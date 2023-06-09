import glob
import json
import os


class IC15_Converter(object):
    """
    Format annotation to standard form for ic15 dataset.
    """

    def __init__(self, path_mode="relative"):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        if task == "rec":
            self._format_rec_label(self.label_path, output_path)

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        with open(output_path, "w") as out_file:
            for label_fp in label_paths:
                label_file_name = os.path.basename(label_fp)
                img_path = os.path.join(image_dir, label_file_name[3:-4] + ".jpg")
                assert os.path.exists(
                    img_path
                ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                label = []
                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)
                with open(label_fp, "r", encoding="utf-8-sig") as f:
                    for line in f.readlines():
                        tmp = line.strip("\n\r").replace("\xef\xbb\xbf", "").split(",")
                        points = tmp[:8]
                        s = []
                        for i in range(0, len(points), 2):
                            b = points[i : i + 2]
                            b = [int(t) for t in b]
                            s.append(b)
                        result = {"transcription": tmp[8], "points": s}
                        label.append(result)

                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")

    def _format_rec_label(self, label_path, output_path):
        with open(output_path, "w") as outf:
            with open(label_path, "r") as f:
                for line in f:
                    # , may occur in text
                    sep_index = line.find(",")
                    img_path = line[:sep_index].strip().replace("\ufeff", "")
                    label = line[sep_index + 1 :].strip().replace('"', "")
                    outf.write(img_path + "\t" + label + "\n")
