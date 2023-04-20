'''
This SVT converter is specifically for the data preparation of SVT dataset (http://www.iapr-tc11.org/mediawiki/index.php/The_Street_View_Text_Dataset)
Please download the data from the website above and unzip the file.
After unzipping the file, the data structure should be like:

svt1
 ├── img
 │   ├── 00_00.jpg 
 │   ├── 00_01.jpg
 │   ├── 00_02.jpg
 │   ├── 00_03.jpg
 │   ├── ...
 ├── test.xml
 └── train.xml

For prepare the data for text recognition, you can run the following command:
python tools/dataset_converters/convert.py \
    --dataset_name  svt --task rec \
    --image_dir path/to/svt1/ \
    --label_dir path/to/svt1/train.xml \
    --output_path path/to/svt1/rec_train_gt.txt 
    
Then you can have a folder `cropped_images/` and an annotation file `rec_train_gt.txt` under the folder `svt1/`.

'''

import os
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np


def xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    imgs_labels = []

    for ch in root:
        im_label = {}
        for ch01 in ch:
            if ch01.tag in "address":
                continue
            elif ch01.tag in 'taggedRectangles':
                # multiple children
                rect_list = []
                for ch02 in ch01:
                    rect = {}
                    rect['location'] = ch02.attrib
                    rect['label'] = ch02[0].text
                    rect_list.append(rect)
                im_label['rect'] = rect_list
            else:
                im_label[ch01.tag] = ch01.text
        imgs_labels.append(im_label)

    return imgs_labels


def image_crop_save(image, location, output_dir):
    """
    crop image with location (h,w,x,y)
    save cropped image to output directory
    """
    start_x = location[2]
    end_x = start_x + location[1]
    start_y = location[3]
    if start_y < 0:
        start_y = 0
    end_y = start_y + location[0]
    print("image array shape :{}".format(image.shape))
    print("crop region ", start_x, end_x, start_y, end_y)
    if len(image.shape) == 3:
        cropped = image[start_y:end_y, start_x:end_x, :]
    else:
        cropped = image[start_y:end_y, start_x:end_x]
    im = Image.fromarray(np.uint8(cropped))
    im.save(output_dir)


class SVT_Converter():
    '''
    Format annotation to standard form for SVT dataset
    '''
    def __init__(self, path_mode='relative'):
        self.path_mode = path_mode
        
    def convert(self, task='rec', image_dir=None, label_path=None, output_path=None):
        assert os.path.exists(label_path), f'{label_path} no exist!'
        
        if task == 'det':
            self._format_det_label(image_dir, label_path, output_path)
        if task == 'rec':
            self._format_rec_label(image_dir, label_path, output_path)

    def _format_det_label(self, image_dir, label_dir, output_path):
        raise NotImplementedError("format det labels is still under development.")
    
    def _format_rec_label(self, image_dir, label_path, output_path):
        if not os.path.exists(image_dir):
            raise ValueError("image_dir :{} does not exist".format(image_dir))

        if not os.path.exists(label_path):
            raise ValueError("xml_file :{} does not exist".format(label_path))

        # new a folder to save cropped images
        new_image_folder_name = "cropped_images"
        root_dir = '/'.join(output_path.split('/')[:-1])
        new_image_dir = os.path.join(root_dir, new_image_folder_name)
        os.makedirs(new_image_dir, exist_ok=True)

        ims_labels_dict = xml_to_dict(label_path)
        num_images = len(ims_labels_dict)
        annotation_list = []
        print("Converting annotation, {} images in total ".format(num_images))
        for i in range(num_images):
            img_label = ims_labels_dict[i]
            image_name = img_label['imageName']
            rects = img_label['rect']
            name, ext = image_name.split('.')
            name = "/".join([new_image_folder_name] + name.split("/")[1:])
            fullpath = os.path.join(image_dir, image_name)
            im_array = np.asarray(Image.open(fullpath))
            print("processing image: {}".format(image_name))
            for j, rect in enumerate(rects):
                rect = rects[j]
                location = rect['location']
                h = int(location['height'])
                w = int(location['width'])
                x = int(location['x'])
                y = int(location['y'])
                label = rect['label']
                loc = [h, w, x, y]
                output_name = name + "_" + str(j) + '.' + ext
                output_file = os.path.join(root_dir, output_name)
                image_crop_save(im_array, loc, output_file)
                ann = output_name + "\t" + label
                annotation_list.append(ann)

        with open(output_path, 'w') as f:
            f.write("\n".join(annotation_list))
