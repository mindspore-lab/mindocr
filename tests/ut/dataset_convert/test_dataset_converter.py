"""
The test ensures that the method correctly formats the label according to the input image directory
and produces the expected output file.
This test sets up a temporary test directory, creates test label and image files, initializes an testing for
ic15 converter.

Example

python tests/convert/test_dataset_converter.py --data_dir ../ocr_datasets
"""

import os
import shutil
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../..")))
import json
import unittest

from tools.dataset_converters.convert import convert, supported_datasets


class TestDatasetPaths(object):
    """
    Test the correctness of the subfolders in the given data directory
    """

    def testAllFolders(self, data_dir):
        self.data_dir = data_dir
        self.checkDirValidity(self.data_dir)
        self.checkDatasetExistence(self.data_dir)
        for d_name in self.dataset_existences:
            eval("self.check" + d_name.upper() + "()")

    def checkDirValidity(self, data_dir):
        assert type(data_dir) is str, f"Expect to get string path, but got {type(data_dir)}"
        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} does not exist!")
        if len(os.listdir(data_dir)) == 0:
            raise ValueError(f"{data_dir} is empty!")

    def checkFileValidity(self, file_path):
        assert type(file_path) is str, f"Expect to get string path, but got {type(file_path)}"
        if not os.path.exists(file_path):
            raise ValueError(f"{file_path} does not exist!")

    def checkDatasetExistence(self, data_dir):
        subfolders = os.listdir(data_dir)
        subfolders = [x for x in subfolders if os.path.isdir(os.path.join(data_dir, x))]
        self.dataset_existences = {}
        for d_name in supported_datasets:
            if d_name in subfolders:
                self.dataset_existences[d_name] = 1

    def checkIC15(self):
        """
        path-to-data-dir/
        ic15/
            ch4_test_images/
            ch4_test_vocabularies_per_image/
            ch4_test_vocabulary.txt
            ch4_training_images/
            ch4_training_localization_transcription_gt/
            ch4_training_vocabularies_per_image/
            ch4_training_vocabulary.txt
            Challenge4_Test_Task4_GT/
            GenericVocabulary.txt
            ch4_test_word_images_gt/
            ch4_training_word_images_gt/
            Challenge4_Test_Task3_GT/
        """
        dataset_dir = os.path.join(self.data_dir, "ic15")
        self.checkDirValidity(dataset_dir)
        for dirname in [
            "ch4_test_images",
            "ch4_test_vocabularies_per_image",
            "ch4_training_images",
            "ch4_training_vocabularies_per_image",
            "ch4_test_word_images_gt",
            "ch4_training_word_images_gt",
            "ch4_training_localization_transcription_gt",
            "Challenge4_Test_Task4_GT",
        ]:
            subfolder = os.path.join(dataset_dir, dirname)
            self.checkDirValidity(subfolder)

        for filename in [
            "ch4_test_vocabulary.txt",
            "ch4_training_vocabulary.txt",
            "GenericVocabulary.txt",
            "Challenge4_Test_Task3_GT.txt",
        ]:
            file_path = os.path.join(dataset_dir, filename)
            self.checkFileValidity(file_path)

    def checkMLT2017(self):
        """
        path-to-data-dir/
            mlt2017/
                Images/
        """
        dataset_dir = os.path.join(self.data_dir, "mlt2017")
        self.checkDirValidity(dataset_dir)
        self.checkDirValidity(os.path.join(dataset_dir, "images/MLT_train_images"))
        self.checkFileValidity(os.path.join(dataset_dir, "train.json"))

    def checkSYNTEXT150K(self):
        """
        path-to-data-dir/
            syntext150k/
                syntext1/
                    images/
                        ecms_imgs/
                    annotations/
                        ecms_v1_maxlen25.json
                syntext2/
                    images/
                        syntext_word_eng/
                    annotations/
                        syntext_word_eng.json

        """
        dataset_dir = os.path.join(self.data_dir, "syntext150k")
        self.checkDirValidity(dataset_dir)
        self.checkDirValidity(os.path.join(dataset_dir, "syntext1/images/emcs_imgs"))
        self.checkDirValidity(os.path.join(dataset_dir, "syntext2/images/syntext_word_eng"))

        self.checkFileValidity(os.path.join(dataset_dir, "syntext1/annotations/ecms_v1_maxlen25.json"))
        self.checkFileValidity(os.path.join(dataset_dir, "syntext2/annotations/syntext_word_eng.json"))

    def checkTOTALTEXT(self):
        """
        path-to-data-dir/
            totaltext/
                images/
                    Test/
                    Train/
                annotations/
                    Test/
                    Train/
        """

        dataset_dir = os.path.join(self.data_dir, "totaltext")
        self.checkDirValidity(dataset_dir)
        self.checkDirValidity(os.path.join(dataset_dir, "Images/Train"))
        self.checkDirValidity(os.path.join(dataset_dir, "Images/Test"))
        self.checkDirValidity(os.path.join(dataset_dir, "annotations/Train"))
        self.checkDirValidity(os.path.join(dataset_dir, "annotations/Test"))


class TestConverterIC15(unittest.TestCase):
    def setUp(self):
        # Create temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_dir")
        os.mkdir(self.test_dir)

        annot_dir = os.path.join(self.test_dir, "annotations")
        os.mkdir(annot_dir)
        self.annot_dir = annot_dir

        # Create label and image files for testing
        label_file = os.path.join(annot_dir, "gt_img1.txt")
        with open(label_file, "w") as f:
            f.write(
                # 8 coordinates: x1, y1, x2, y2, x3, y3, x4, y4, transcription
                "377,117,463,117,465,130,378,130,Genaxis Theatre\n"
                "374,155,409,155,409,170,374,170,###\n"
            )
        label_file = os.path.join(annot_dir, "gt_img2.txt")
        with open(label_file, "w") as f:
            f.write("602,173,635,175,634,197,602,196,EXIT\n" "734,310,792,320,792,364,738,361,I2R\n")
        self.image_dir = os.path.join(self.test_dir, "test_images")
        os.mkdir(self.image_dir)
        img1_path = os.path.join(self.image_dir, "img1.jpg")
        img2_path = os.path.join(self.image_dir, "img2.jpg")
        open(img1_path, "a").close()  # dummy image path
        open(img2_path, "a").close()  # dummpy image path

    def tearDown(self):
        # Remove temporary test directory
        shutil.rmtree(self.test_dir)

    def testIC15_Converter(self):
        convert("ic15", "det", self.image_dir, self.annot_dir, output_path=None, path_mode="relative")
        self._test_format_det_label()

    def _test_format_det_label(self):
        # Convert labels
        output_file = os.path.join(self.test_dir, "det_gt.txt")

        # Check that output file was created with the expected content
        with open(output_file, "r") as f:
            contents = f.read()
        expected_contents = (
            'img1.jpg\t[{"transcription": "Genaxis Theatre", "points": [[377, 117], [463, 117], [465, 130], '
            '[378, 130]]}, {"transcription": "###", "points": [[374, 155], [409, 155], [409, 170], [374, 170]]}]\n'
            'img2.jpg\t[{"transcription": "EXIT", "points": [[602, 173], [635, 175], [634, 197], [602, 196]]}, '
            '{"transcription": "I2R", "points": [[734, 310], [792, 320], [792, 364], [738, 361]]}]\n'
        )
        self.assertEqual(contents, expected_contents)


class TestConverterTOTALTEXT(unittest.TestCase):
    def setUp(self):
        # Create temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_dir")
        os.mkdir(self.test_dir)

        annot_dir = os.path.join(self.test_dir, "annotations")
        os.mkdir(annot_dir)
        self.annot_dir = annot_dir

        # Create label and image files for testing
        label_file = os.path.join(annot_dir, "poly_gt_img1.txt")
        with open(label_file, "w") as f:
            f.write(
                "x: [[206 251 386 542 620 646 550 358 189 140]], "
                "y: [[ 633  811  931  946  926  976 1009  989  845  629]], "
                "ornt: [u'c'], transcriptions: [u'PETROSAINS']\n"
            )  # 20 coordinates, orientations, transcription
        label_file = os.path.join(annot_dir, "poly_gt_img2.txt")
        with open(label_file, "w") as f:
            f.write(
                # 8 coordinates, orientations, transcription
                "x: [[115 503 494 115]], y: [[322 346 426 404]], ornt: [u'm'], transcriptions: [u'nauGHTY']\n"
                "x: [[ 734 1058 1061  744]], y: [[360 369 449 430]], ornt: [u'm'], transcriptions: [u'NURIS']\n"
            )
        self.image_dir = os.path.join(self.test_dir, "test_images")
        os.mkdir(self.image_dir)
        img1_path = os.path.join(self.image_dir, "img1.jpg")
        img2_path = os.path.join(self.image_dir, "img2.jpg")
        open(img1_path, "a").close()  # dummy image path
        open(img2_path, "a").close()  # dummpy image path

    def tearDown(self):
        # Remove temporary test directory
        shutil.rmtree(self.test_dir)

    def testTOTALTEXT_Converter(self):
        convert("totaltext", "det", self.image_dir, self.annot_dir, output_path=None, path_mode="relative")
        self._test_format_det_label()

    def _test_format_det_label(self):
        # Convert labels
        output_file = os.path.join(self.test_dir, "det_gt.txt")

        # Check that output file was created with the expected content
        with open(output_file, "r") as f:
            contents = f.read()
        expected_contents = (
            'img1.jpg\t[{"transcription": "PETROSAINS", "points": [[206, 633], [251, 811], [386, 931], [542, 946], '
            "[620, 926], [646, 976], [550, 1009], [358, 989], [189, 845], [140, 629]]}]\n"
            'img2.jpg\t[{"transcription": "nauGHTY", "points": [[115, 322], [503, 346], [494, 426], [115, 404]]}, '
            '{"transcription": "NURIS", "points": [[734, 360], [1058, 369], [1061, 449], [744, 430]]}]\n'
        )
        self.assertEqual(contents, expected_contents)


class TestConverterCOCOFORMAT(unittest.TestCase):
    def setUp(self):
        # Create temporary test directory
        self.test_dir = os.path.join(os.getcwd(), "test_dir")
        os.mkdir(self.test_dir)

        self.label_file = os.path.join(self.test_dir, "train.json")

        # Create label and image files for testing
        label_img1 = [
            {
                "image_id": 60001,
                "bbox": [218.0, 406.0, 138.0, 47.0],
                "area": 6486.0,
                "rec": [
                    95,
                    95,
                    95,
                    95,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                ],
                "category_id": 1,
                "iscrowd": 0,
                "id": 1,
                "bezier_pts": [
                    219.0,
                    406.0,
                    264.33,
                    407.0,
                    309.67,
                    408.0,
                    355.0,
                    409.0,
                    352.0,
                    448.0,
                    307.33,
                    449.33,
                    262.67,
                    450.67,
                    218.0,
                    452.0,
                ],
            },
        ]
        data_img1 = {
            "width": 400,
            "date_captured": "",
            "license": 0,
            "flickr_url": "",
            "file_name": "0060001.jpg",
            "id": 60001,
            "coco_url": "",
            "height": 600,
        }
        label_img2 = [
            {
                "image_id": 60002,
                "bbox": [335.0, 81.0, 252.0, 110.0],
                "area": 27720.0,
                "rec": [
                    95,
                    95,
                    95,
                    95,
                    95,
                    95,
                    95,
                    95,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                ],
                "category_id": 1,
                "iscrowd": 0,
                "id": 6,
                "bezier_pts": [
                    341.0,
                    81.0,
                    422.67,
                    98.33,
                    504.33,
                    115.67,
                    586.0,
                    133.0,
                    582.0,
                    190.0,
                    499.67,
                    172.67,
                    417.33,
                    155.33,
                    335.0,
                    138.0,
                ],
            },
            {
                "image_id": 60002,
                "bbox": [69.0, 182.0, 17.0, 19.0],
                "area": 323.0,
                "rec": [
                    20,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                    96,
                ],
                "category_id": 1,
                "iscrowd": 0,
                "id": 7,
                "bezier_pts": [
                    70.0,
                    183.0,
                    75.0,
                    182.67,
                    80.0,
                    182.33,
                    85.0,
                    182.0,
                    84.0,
                    199.0,
                    79.0,
                    199.33,
                    74.0,
                    199.67,
                    69.0,
                    200.0,
                ],
            },
        ]
        data_img2 = {
            "width": 400,
            "date_captured": "",
            "license": 0,
            "flickr_url": "",
            "file_name": "0060002.jpg",
            "id": 60002,
            "coco_url": "",
            "height": 600,
        }
        label_file = {
            "licenses": [],
            "info": {},
            "categories": [
                {
                    "supercategory": "beverage",
                    "id": 1,
                    "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
                    "name": "text",
                }
            ],
        }
        label_file["annotations"] = label_img1 + label_img2
        label_file["images"] = [data_img1, data_img2]
        with open(self.label_file, "w") as file:
            json.dump(label_file, file)

        self.image_dir = os.path.join(self.test_dir, "test_images")
        os.mkdir(self.image_dir)
        img1_path = os.path.join(self.image_dir, "0060001.jpg")
        img2_path = os.path.join(self.image_dir, "0060002.jpg")
        open(img1_path, "a").close()  # dummy image path
        open(img2_path, "a").close()  # dummpy image path

    def tearDown(self):
        # Remove temporary test directory
        shutil.rmtree(self.test_dir)

    def testMLT2017_Converter(self):
        convert("mlt2017", "det", self.image_dir, self.label_file, output_path=None, path_mode="relative")
        self._test_format_det_label()

    def testSYNTEXT150K_Converter(self):
        convert("syntext150k", "det", self.image_dir, self.label_file, output_path=None, path_mode="relative")
        self._test_format_det_label()

    def _test_format_det_label(self):
        # Convert labels
        output_file = os.path.join(self.test_dir, "det_gt.txt")

        # Check that output file was created with the expected content
        with open(output_file, "r") as f:
            contents = f.read()
        expected_contents = (
            '0060001.jpg\t[{"transcription": "口口口口", "points": [[218, 406], [356, 406], [356, 453], [218, 453]], '
            '"bezier": [219.0, 406.0, 264.33, 407.0, 309.67, 408.0, 355.0, 409.0, 352.0, 448.0, 307.33, 449.33, '
            "262.67, 450.67, 218.0, 452.0]}]\n"
            '0060002.jpg\t[{"transcription": "口口口口口口口口", "points": [[335, 81], [587, 81], [587, 191], [335, 191]], '
            '"bezier": [341.0, 81.0, 422.67, 98.33, 504.33, 115.67, 586.0, 133.0, 582.0, 190.0, 499.67, 172.67, '
            '417.33, 155.33, 335.0, 138.0]}, {"transcription": "4", "points": [[69, 182], [86, 182], [86, 201], '
            '[69, 201]], "bezier": [70.0, 183.0, 75.0, 182.67, 80.0, 182.33, 85.0, 182.0, 84.0, 199.0, 79.0, 199.33, '
            "74.0, 199.67, 69.0, 200.0]}]\n"
        )
        self.assertEqual(contents, expected_contents)


if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="input dir")

    # Add some arguments to the parser
    parser.add_argument("--data_dir", type=str, help="a directory containing all datasets", default=None)
    # Parse the arguments
    args = parser.parse_args()
    if args.data_dir is not None:
        TestDatasetPaths().testAllFolders(args.data_dir)

    suite = unittest.TestSuite()
    suite.addTest(TestConverterIC15("testIC15_Converter"))
    suite.addTest(TestConverterTOTALTEXT("testTOTALTEXT_Converter"))

    suite.addTest(TestConverterCOCOFORMAT("testMLT2017_Converter"))
    suite.addTest(TestConverterCOCOFORMAT("testSYNTEXT150K_Converter"))

    runner = unittest.TextTestRunner()
    runner.run(suite)
