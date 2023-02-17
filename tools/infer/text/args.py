import argparse


def get_args():
    parser = argparse.ArgumentParser()


    # mxocr
    parser.add_argument("--input_images_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--parallel_num", type=str, required=False, default=1)
    parser.add_argument("--infer_res_save_path", type=str, required=False, default=None)

    return parser.parse_args()
