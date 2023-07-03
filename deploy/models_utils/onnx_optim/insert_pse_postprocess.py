import argparse
import os

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.helper import make_node


def add_pse_nodes(node, binary_thresh, scale):
    input_name = node[-1].output[0]
    extended_op_nodes = [
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.14"],
            value=numpy_helper.from_array(
                np.array([1.0, 1.0, 4.0 / scale, 4.0 / scale], dtype="float32"), name="p2o.helper.constant.14"
            ),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.15"],
            value=numpy_helper.from_array(np.array([], dtype="float32"), name="p2o.helper.constant.15"),
        ),
        make_node(
            "Resize",
            inputs=[input_name, "p2o.helper.constant.15", "p2o.helper.constant.14"],
            outputs=["bilinear_interp_v2_6.tmp_0"],
            name="p2o.Resize.6",
            mode="linear",
            coordinate_transformation_mode="half_pixel",
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.16"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.16"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.17"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.17"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.18"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.18"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.19"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.19"),
        ),
        make_node(
            "Slice",
            inputs=[
                "bilinear_interp_v2_6.tmp_0",
                "p2o.helper.constant.16",
                "p2o.helper.constant.17",
                "p2o.helper.constant.19",
                "p2o.helper.constant.18",
            ],
            outputs=["p2o.Slice.1"],
            name="p2o.Slice.0",
        ),
        make_node(
            "Squeeze",
            inputs=["p2o.Slice.1"],
            outputs=["bilinear_interp_v2_6.tmp_0_slice_0"],
            name="p2o.Squeeze.0",
            axes=[1],
        ),
        make_node(
            "Sigmoid", inputs=["bilinear_interp_v2_6.tmp_0_slice_0"], outputs=["sigmoid_0.tmp_0"], name="p2o.Sigmoid.0"
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["tmp_3"],
            value=numpy_helper.from_array(np.array([binary_thresh], dtype="float32"), name="tmp_3"),
        ),
        make_node("Greater", inputs=["bilinear_interp_v2_6.tmp_0", "tmp_3"], outputs=["tmp_4"], name="p2o.Greater.0"),
        make_node("Cast", inputs=["tmp_4"], outputs=["tmp_5"], name="p2o.Cast.0", to=TensorProto.FLOAT),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.20"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.20"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.21"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.21"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.22"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.22"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.23"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.23"),
        ),
        make_node(
            "Slice",
            inputs=[
                "tmp_5",
                "p2o.helper.constant.20",
                "p2o.helper.constant.21",
                "p2o.helper.constant.23",
                "p2o.helper.constant.22",
            ],
            outputs=["p2o.Slice.3"],
            name="p2o.Slice.2",
        ),
        make_node("Squeeze", inputs=["p2o.Slice.3"], outputs=["tmp_5_slice_0"], name="p2o.Squeeze.1", axes=[1]),
        make_node(
            "Unsqueeze", inputs=["tmp_5_slice_0"], outputs=["unsqueeze2_0.tmp_0"], name="p2o.Unsqueeze.0", axes=[1]
        ),
        make_node("Mul", inputs=["tmp_5", "unsqueeze2_0.tmp_0"], outputs=["tmp_6"], name="p2o.Mul.0"),
    ]
    node.extend(extended_op_nodes)


# python insert_pse_postprocess.py --model_path ./pse_r50vd.onnx
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--binary_thresh", type=float, required=False, default=0.0)
    parser.add_argument("--scale", type=float, required=False, default=1.0)
    args = parser.parse_args()

    model = onnx.load(args.model_path)
    graph = model.graph
    node = graph.node
    result_path = os.path.dirname(args.model_path)
    basename, ext = os.path.splitext(os.path.basename(args.model_path))

    add_pse_nodes(node, args.binary_thresh, args.scale)

    # modify output
    graph.output.pop()  # delete original output
    graph.output.extend(
        [
            onnx.helper.make_tensor_value_info(
                "sigmoid_0.tmp_0",
                TensorProto.FLOAT,
                shape=["p2o.DynamicDimension.6", "p2o.DynamicDimension.7", "p2o.DynamicDimension.8"],
            ),
            onnx.helper.make_tensor_value_info(
                "tmp_6",
                TensorProto.FLOAT,
                shape=["p2o.DynamicDimension.3", 7, "p2o.DynamicDimension.4", "p2o.DynamicDimension.5"],
            ),
        ]
    )

    save_name = basename + "_pse_post_new" + ext
    save_path = os.path.join(result_path, save_name)
    onnx.save(model, save_path)

    print(f"The new model saved in {save_path}")
