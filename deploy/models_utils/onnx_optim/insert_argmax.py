import argparse
import ast
import math
import os

import cv2
import numpy as np
import onnx
import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument


def resize_norm_img(img, max_wh_ratio, img_c, img_h):
    img_w = int(img_h * max_wh_ratio)

    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(img_h * ratio) > img_w:
        resized_w = img_w
    else:
        resized_w = math.ceil(img_h * ratio)
    resized_image = cv2.resize(img, (resized_w, img_h))
    resized_image = resized_image.astype("float32")
    if len(resized_image.shape) < 3:
        resized_image = np.expand_dims(resized_image, 2)
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--check_output_onnx",
        type=ast.literal_eval,
        required=False,
        default=True,
        choices=[True, False],
    )
    args = parser.parse_args()

    model = onnx.load(args.model_path)
    graph = model.graph
    node = graph.node
    print(f"Input Channel of Net: {graph.input[0].type.tensor_type.shape.dim[1].dim_value}")
    print(f"Input Height of Net: {graph.input[0].type.tensor_type.shape.dim[2].dim_value}")
    result_path = os.path.dirname(args.model_path)
    basename, ext = os.path.splitext(os.path.basename(args.model_path))
    img_c = graph.input[0].type.tensor_type.shape.dim[1].dim_value
    img_h = graph.input[0].type.tensor_type.shape.dim[2].dim_value
    img_h = 32 if img_h == -1 else img_h
    output_name = "argmax_0.tmp_0"
    input_name = ""
    insert_index = 0
    for i in range(len(node)):
        if node[i].op_type == "Softmax":
            input_name = node[i].output[0]
            insert_index = i
        if node[i].op_type == "ArgMax":
            raise ValueError(
                "ArgMax Op found. The model already has ArgMax Op. Please check the type of model. Input "
                "model type should be CRNN or SVTR."
            )

    if not input_name:
        raise ValueError(
            "Softmax Op not found. Please check the type of the model. Input model type should be CRNN or SVTR."
        )
    axis = 2
    keepdims = 0
    argmax_operator = onnx.helper.make_node(
        "ArgMax",
        inputs=[input_name],
        outputs=[output_name],
        name="ArgMax_0",
        axis=axis,
        keepdims=keepdims,
    )

    node.insert(insert_index, argmax_operator)
    if len(model.opset_import) > 1:
        del model.opset_import[1]

    graph.output[0].type.tensor_type.elem_type = 7
    graph.output[0].type.tensor_type.shape.dim[0].dim_value = -1
    graph.output[0].type.tensor_type.shape.dim[1].dim_value = -1
    graph.output[0].name = output_name
    graph.output[0].type.tensor_type.shape.dim.pop()

    save_name = basename + "_argmax" + ext
    save_path = os.path.join(result_path, save_name)
    onnx.save(model, save_path)

    if args.check_output_onnx:
        np.random.seed(2022)
        image = 255 * np.random.randn(74, 960, img_c)
        h, w = image.shape[:2]
        max_wh_ratio = w * 1.0 / h

        image = resize_norm_img(image, max_wh_ratio, img_c, img_h)
        image = np.reshape(image, ((1,) + image.shape[:3]))
        try:
            session = onnxruntime.InferenceSession(save_path)
            session.get_modelmeta()
            output2 = session.run(["argmax_0.tmp_0"], {"x": image})
        except (RuntimeError, InvalidArgument) as error:
            print("------------------------------------------------------------------------------------")
            print("onnx check failed. Please check the error Message and contact the support engineer.")
            print("------------------------------------------------------------------------------------")
            raise error
        else:
            print(f"onnx check pass. The new model saved in {save_path}")
    else:
        print(f"The new model saved in {save_path}")
