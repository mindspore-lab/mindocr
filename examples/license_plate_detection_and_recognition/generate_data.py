import json
import os

from PIL import Image


def read_annotations(file_path):
    annotations = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            image_name, description = line.strip().split("\t")
            description_data = json.loads(description)
            annotations.append((image_name, description_data))
    return annotations


def crop_images(annotations, source_folder, target_folder, output_txt):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    with open(output_txt, "w", encoding="utf-8") as out_file:
        for image_name, data in annotations:
            image_path = os.path.join(source_folder, image_name)
            with Image.open(image_path) as img:
                bbox = data[0]["bbox"]
                x1, y1 = bbox[0]
                x2, y2 = bbox[1]
                cropped_img = img.crop((x1, y1, x2, y2))
                cropped_image_name = f"{image_name}"
                cropped_img.save(os.path.join(target_folder, cropped_image_name))
                transcription = data[0]["transcription"]
                out_file.write(f"{cropped_image_name}\t{transcription}\n")


def main():
    datasets = ["train", "test", "val"]
    for dataset in datasets:
        annotations_file = f"path/to/DBNet_DataSets/{dataset}/{dataset}_det_gt.txt"
        source_folder = f"path/to/DBNet_DataSets/{dataset}/images"
        target_folder = f"path/to/SVTR_DataSets/{dataset}/"
        output_txt = f"path/to/SVTR_DataSets/gt_{dataset}.txt"
        annotations = read_annotations(annotations_file)
        crop_images(annotations, source_folder, target_folder, output_txt)


if __name__ == "__main__":
    main()
