import json
import os

import tqdm


class SYNTEXT150K_Converter(object):
    """
    Format annotation to standard form for SYNTEXT150K dataset. The original annotation file is a COCO-format JSON
    annotation file.
    When loaded with json library, it is a dictionary data with following keys:
    dict_keys(['licenses', 'info', 'images', 'annotations', 'categories'])
    An example of data['images'] (a list of dictionaries):
    {'width': 400, 'date_captured': '', 'license': 0, 'flickr_url': '', 'file_name': '0000000.jpg', 'id': 60001,
    'coco_url': '', 'height': 600}
    An example of data['annotations'] (a list of dictionaries):
    {'image_id': 60001, 'bbox': [218.0, 406.0, 138.0, 47.0], 'area': 6486.0, 'rec': [95, ..., 96], 'category_id': 1,
    'iscrowd': 0, 'id': 1, 'bezier_pts': [219.0, ..., 218.0, 452.0]}
    'bbox' is defined by [x_min, y_min, width, height] in coco format.

    self._format_det_label transforms the annotations into a single det label file with a format like:
    0000000.jpg	[{"transcription": "the", "points":[[153, 347], ..., [177, 357]], 'beizer':[123,...,567]}]
    """

    def __init__(self, path_mode="relative"):
        self.path_mode = path_mode
        self.CTLABELS = [
            " ",
            "!",
            '"',
            "#",
            "$",
            "%",
            "&",
            "'",
            "(",
            ")",
            "*",
            "+",
            ",",
            "-",
            ".",
            "/",
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
            ":",
            ";",
            "<",
            "=",
            ">",
            "?",
            "@",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
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
            "[",
            "\\",
            "]",
            "^",
            "_",
            "`",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "{",
            "|",
            "}",
            "~",
        ]
        self.vocabulary_size = len(self.CTLABELS) + 1

    def convert(self, task="det", image_dir=None, label_path=None, output_path=None):
        self.label_path = label_path
        assert os.path.exists(label_path), f"{label_path} no exist!"

        if task == "det":
            self._format_det_label(image_dir, self.label_path, output_path)
        if task == "rec":
            raise ValueError("SynText dataset has no cropped word images and recognition labels.")

    def _decode_rec_ids_to_string(self, rec):
        transcription = ""
        for index in rec:
            if index == self.vocabulary_size - 1:
                transcription += "口"
            elif index < self.vocabulary_size - 1:
                transcription += self.CTLABELS[index]
        return transcription

    def _format_det_label(self, image_dir, label_path, output_path):
        with open(output_path, "w") as out_file:
            coco_json_data = json.load(open(label_path, "r"))
            annotations = coco_json_data["annotations"]
            images_labels = {}
            for annot in tqdm.tqdm(annotations, total=len(annotations)):
                image_id = annot["image_id"]
                img_path = os.path.join(
                    image_dir, "{:07d}".format(image_id) + ".jpg"
                )  # sometimes {:07d} works, sometimes {:08d} works
                if not os.path.exists(img_path):
                    img_path = os.path.join(image_dir, "{:08d}".format(image_id) + ".jpg")
                assert os.path.exists(img_path), f"{img_path} not exist! Please check the input image_dir {image_dir}"
                if self.path_mode == "relative":
                    img_path = os.path.basename(img_path)
                if img_path not in images_labels:
                    images_labels[img_path] = []

                bbox = annot["bbox"]  # [x_min, y_min, width, height]
                bbox = [
                    [bbox[0], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                    [bbox[0], bbox[1] + bbox[3]],
                ]
                bbox = [[int(x[0]), int(x[1])] for x in bbox]
                bezier = annot["bezier_pts"]
                transcription = self._decode_rec_ids_to_string(
                    annot["rec"]
                )  # needs to translate from character ids to characters.
                images_labels[img_path].append({"transcription": transcription, "points": bbox, "bezier": bezier})
            for img_path in images_labels:
                annotations = []
                for annot_instance in images_labels[img_path]:
                    annotations.append(annot_instance)
                out_file.write(img_path + "\t" + json.dumps(annotations, ensure_ascii=False) + "\n")
