import os
import json
from PIL import Image

def read_annotations(file_path):
    annotations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            image_name, description = line.strip().split('\t')
            description_data = json.loads(description)
            annotations.append((image_name, description_data))
    return annotations

def crop_images(annotations, source_folder, target_folder, output_txt):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with open(output_txt, 'w', encoding='utf-8') as out_file:
        for image_name, data in annotations:
            # 构造图片路径
            image_path = os.path.join(source_folder, image_name)
            
            # 打开图像
            with Image.open(image_path) as img:
                # 获取边框
                bbox = data[0]['bbox']  # 假设 bbox 是第一个元素
                x1, y1 = bbox[0]  # 左上角
                x2, y2 = bbox[1]  # 右下角
                
                # 裁剪图像
                cropped_img = img.crop((x1, y1, x2, y2))
                
                # 保存裁剪后的图像
                cropped_image_name = f"{image_name}"
                cropped_img.save(os.path.join(target_folder, cropped_image_name))

                # 写入新的 txt 文件
                transcription = data[0]['transcription']
                out_file.write(f"{cropped_image_name}\t{transcription}\n")

def main():
    # 定义数据集类型
    datasets = ['train', 'test', 'val']


    for dataset in datasets:
        annotations_file = f'path/to/DBNet_DataSets/{dataset}/{dataset}_det_gt.txt'
        source_folder = f'path/to/DBNet_DataSets/{dataset}/images'  # 源图片文件夹
        target_folder = f'path/to/SVTR_DataSets/{dataset}/'  # 目标文件夹
        output_txt = f'path/to/SVTR_DataSets/gt_{dataset}.txt'  # 输出文件

        # 读取标注
        annotations = read_annotations(annotations_file)
        
        # 裁剪图像
        crop_images(annotations, source_folder, target_folder, output_txt)

if __name__ == "__main__":
    main()
