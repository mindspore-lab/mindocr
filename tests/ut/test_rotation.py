"""
This test case is used to test an image rotation tool that rotates the target image 180 degrees

Args:
    img_path: Input image address
    output_path: Output image save address

Return:
    Saves the rotated image to the specified address
"""
import os
import sys
import cv2

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../../")))

from tools.infer.text.utils import img_rotate


def test_img_rotate(img_path, output_path):
    image = cv2.imread(img_path)
    img_180 = img_rotate(image, 180)
    cv2.imwrite(output_path, img_180)


if __name__=='__main__':
    img_path= "path_to_img_file"
    output_path="path_to_save_dir/output.png"
    test_img_rotate(img_path,output_path)
