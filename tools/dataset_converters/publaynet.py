import json
import os

CATEGORIES = {1: 1, 2: 0, 3: 4, 4: 3, 5: 2}


class PUBLAYNET_Converter:
    """
    Format annotation to yolo form for PublayNet dataset.
    """

    def __init__(self, path_mode=None, **kwargs):
        pass

    @staticmethod
    def _summarize_files_into_txt(directory, mode):
        if mode not in ("train", "val"):
            raise ValueError('mode must be "train" or "val", but got {}'.format(mode))
        directory = os.path.join(directory, mode)
        if directory.endswith(os.sep):
            path = directory[:-1] + ".txt"
        else:
            path = directory + ".txt"
        if os.path.isfile(path):
            print("{} already exists.".format(path))
        else:
            listdir = os.listdir(directory)
            data = ""
            for file in listdir:
                data += os.path.join(directory, file) + "\r\n"
            with open(path, "w") as f:
                f.write(data)

    @staticmethod
    def _generate_label_txt(directory, mode):
        if mode not in ("train", "val"):
            raise ValueError('mode must be "train" or "val", but got {}'.format(mode))
        label_path = os.path.join(directory, mode + ".json")
        id_dict = dict()
        if os.path.isfile(label_path):
            with open(label_path, "r") as f:
                data = json.load(f)
        else:
            raise ValueError("{} does not exist.".format(label_path))
        for item in data["images"]:
            id_dict[item["id"]] = (item["file_name"], item["width"], item["height"])
        converted_label_dict = dict()
        for item in data["annotations"]:
            file_name, width, height = id_dict[item["image_id"]]
            file_name = file_name.replace("jpg", "txt")
            if file_name not in converted_label_dict:
                converted_label_dict[file_name] = ""
            x, y, w, h = item["bbox"]
            x_center = x + w * 0.5
            y_center = y + h * 0.5
            x_center /= width
            y_center /= height
            w /= width
            h /= height

            converted_label = "{} {} {} {} {}".format(CATEGORIES[item["category_id"]], x_center, y_center, w, h)
            converted_label_dict[file_name] += converted_label + "\r\n"

        out_dir = os.path.join(directory, mode)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        for file_name in converted_label_dict:
            with open(os.path.join(out_dir, file_name), "w") as f:
                f.write(converted_label_dict[file_name])

    def convert(self, task=None, image_dir=None, label_path=None, output_path=None):
        if not os.path.isdir(image_dir):
            raise ValueError("image_dir is not a directory: {}".format(image_dir))
        listdir = os.listdir(image_dir)
        if "train" not in listdir or "val" not in listdir:
            raise ValueError("train/ and val/ must be in image_dir")
        self._summarize_files_into_txt(image_dir, "train")
        self._summarize_files_into_txt(image_dir, "val")
        self._generate_label_txt(image_dir, "train")
        self._generate_label_txt(image_dir, "val")
