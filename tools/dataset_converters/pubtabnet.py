import json
from pathlib import Path


class PUBTABNET_Converter:
    """
    Format annotations into standard form for pubtabnet dataset.
    """

    def __init__(self, path_mode="relative", **kwargs):
        self._relative = path_mode == "relative"

        if "split" not in kwargs:
            raise ValueError("It is required to specify the `split` argument for the pubtabnet dataset converter.")
        self._split = kwargs["split"]

    def convert(self, task="table", image_dir=None, label_path=None, output_path=None):
        label_path = Path(label_path)
        assert label_path.exists(), f"{label_path} does not exist!"

        if task == "table":
            self._format_table_label(Path(image_dir), label_path, output_path)
        else:
            raise NotImplementedError(f"PUBTABNET does not support task {task}.")

    def _format_table_label(self, image_dir: Path, label_path: Path, output_path: str):
        processed = 0
        with open(output_path, "w", encoding="utf-8") as out_file:
            with open(label_path, "r") as json_file:
                for line in json_file:
                    line = json.loads(line)
                    if line["split"] != self._split:
                        continue

                    img_path = image_dir / line["filename"]
                    assert img_path.exists(), f"Image {img_path} not found."

                    line["filename"] = img_path.name if self._relative else str(img_path)

                    out_file.write(json.dumps(line, ensure_ascii=False) + "\n")
                    processed = processed + 1

        print(f"{processed} labels for {self._split} set are processed.")
