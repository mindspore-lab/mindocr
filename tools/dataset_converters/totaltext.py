import glob
import json
import os
import re


class TOTALTEXT_Converter(object):
    """
    Format annotation to standard form for total text dataset. The original annotation files are txt files named after
        the image names. An example of txt file:
    x: [[153 161 179 195 184 177]], y: [[347 323 305 315 331 357]], ornt: [u'c'], transcriptions: [u'the']
    x: [[184 222 273 269 230 202]], y: [[293 269 270 296 297 317]], ornt: [u'c'], transcriptions: [u'alpaca']

    The self._format_det_label transforms the above annotation into a single line in the output label file:
    img1001.jpg	[{"transcription": "the", "points": [[153, 347], ..., [177, 357]]}, {"transcription": "alpaca",
    "points": [[184, 293], ..., [202, 317]]}]
    ...
    """

    def __init__(self, path_mode="relative"):
        self.path_mode = path_mode

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        if task == "rec":
            raise ValueError("total-text dataset has no cropped word images and recognition labels.")

    def _format_det_label(self, image_dir, label_dir, output_path):
        label_paths = sorted(glob.glob(os.path.join(label_dir, "*.txt")))
        with open(output_path, "w") as out_file:
            for label_fp in label_paths:
                label_file_name = os.path.basename(label_fp)
                img_path = os.path.join(image_dir, label_file_name.split("_")[-1].split(".")[0] + ".jpg")
                if not os.path.exists(img_path):
                    img_path = img_path.replace(".jpg", ".JPG")
                assert os.path.exists(
                    img_path
                ), f"{img_path} not exist! Please check the input image_dir {image_dir} and names in {label_fp}"
                label = []
                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)
                with open(label_fp, "r", encoding="utf-8-sig") as f:
                    line_saver = ""  # sometimes the same text instance is saved in multiple lines, not a single line
                    for line in f.readlines():
                        tmp = line.strip("\n\r").replace("\xef\xbb\xbf", "").split(", ")
                        assert len(tmp), f"parse error for {tmp}."
                        if len(tmp) < 4 or len(line_saver) > 0:
                            line_saver += line
                            new_splits = line_saver.strip("\n\r").replace("\xef\xbb\xbf", "").split(", ")
                            if len(new_splits) < 4:
                                continue
                            elif len(new_splits) == 4:
                                tmp = new_splits
                                line_saver = ""
                            else:
                                raise ValueError("Parse Error {line_saver}")
                        xs, ys, _, transcriptions = tmp
                        xs = re.findall(r"\S+", xs.split(": ")[-1].replace("[", "").replace("]", ""))
                        ys = re.findall(r"\S+", ys.split(": ")[-1].replace("[", "").replace("]", ""))
                        transcriptions = (
                            transcriptions.split(":")[-1].replace("[", "").replace("]", "").strip().split(" ")
                        )  # transcription looks like [u'xxx']
                        assert len(transcriptions) == 1, f"parse transcription {transcriptions} error."
                        transcriptions = transcriptions[0]
                        if transcriptions.startswith("u"):
                            transcriptions = transcriptions[1:]
                        transcriptions = transcriptions.replace("'", "")
                        s = []
                        for x, y in zip(xs, ys):
                            if len(x) > 0 and len(y) > 0:
                                b = [int(x), int(y)]
                                s.append(b)
                        result = {"transcription": transcriptions, "points": s}
                        label.append(result)

                out_file.write(img_path + "\t" + json.dumps(label, ensure_ascii=False) + "\n")

    # def _format_rec_label(self, label_path, output_path):
    # with open(output_path, 'w') as outf:
    #     with open(label_path, 'r') as f:
    #         for line in f:
    #             # , may occur in text
    #             sep_index = line.find(',')
    #             img_path = line[:sep_index].strip().replace('\ufeff', '')
    #             label = line[sep_index+1:].strip().replace("\"", "")
    #             outf.write(img_path + '\t' + label + '\n')
