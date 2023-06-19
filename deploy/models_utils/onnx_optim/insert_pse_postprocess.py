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
                np.array([1.0, 1.0, 4 / scale, 4 / scale], dtype="float32"), name="p2o.helper.constant.14"
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
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.24"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.24"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.25"],
            value=numpy_helper.from_array(np.array([7], dtype="int64"), name="p2o.helper.constant.25"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.26"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.26"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.27"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.27"),
        ),
        make_node(
            "Slice",
            inputs=[
                "tmp_5",
                "p2o.helper.constant.24",
                "p2o.helper.constant.25",
                "p2o.helper.constant.27",
                "p2o.helper.constant.26",
            ],
            outputs=["tmp_5_slice_1"],
            name="p2o.Slice.4",
        ),
        make_node("Mul", inputs=["tmp_5_slice_1", "unsqueeze2_0.tmp_0"], outputs=["p2o.Mul.1"], name="p2o.Mul.0"),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.28"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.28"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.29"],
            value=numpy_helper.from_array(np.array([2147483647], dtype="int64"), name="p2o.helper.constant.29"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.30"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.30"),
        ),
        make_node("Shape", inputs=["tmp_5"], outputs=["p2o.Shape.1"], name="p2o.Shape.0"),
        make_node(
            "Gather",
            inputs=["p2o.Shape.1", "p2o.helper.constant.30"],
            outputs=["p2o.Gather.1"],
            name="p2o.Gather.0",
            axis=0,
        ),
        make_node("Min", inputs=["p2o.Gather.1", "p2o.helper.constant.29"], outputs=["p2o.Min.1"], name="p2o.Min.0"),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.31"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.31"),
        ),
        make_node(
            "Slice",
            inputs=["tmp_5", "p2o.helper.constant.28", "p2o.Min.1", "p2o.helper.constant.30", "p2o.helper.constant.31"],
            outputs=["p2o.Slice.6"],
            name="p2o.Slice.5",
        ),
        make_node("Shape", inputs=["p2o.Slice.6"], outputs=["p2o.Shape.3"], name="p2o.Shape.2"),
        make_node("Expand", inputs=["p2o.Mul.1", "p2o.Shape.3"], outputs=["p2o.Expand.1"], name="p2o.Expand.0"),
        make_node("Squeeze", inputs=["p2o.helper.constant.28"], outputs=["p2o.helper.squeeze.0"], name="p2o.Squeeze.2"),
        make_node("Squeeze", inputs=["p2o.Min.1"], outputs=["p2o.helper.squeeze.1"], name="p2o.Squeeze.3"),
        make_node("Squeeze", inputs=["p2o.helper.constant.31"], outputs=["p2o.helper.squeeze.2"], name="p2o.Squeeze.4"),
        make_node(
            "Range",
            inputs=["p2o.helper.squeeze.0", "p2o.helper.squeeze.1", "p2o.helper.squeeze.2"],
            outputs=["p2o.Range.1"],
            name="p2o.Range.0",
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.32"],
            value=numpy_helper.from_array(np.array([1, -1, 1, 1], dtype="int64"), name="p2o.helper.constant.32"),
        ),
        make_node(
            "Reshape",
            inputs=["p2o.Range.1", "p2o.helper.constant.32"],
            outputs=["p2o.helper.reshape.0"],
            name="p2o.Reshape.4",
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.33"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.33"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.34"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.34"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.35"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.35"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.36"],
            value=numpy_helper.from_array(np.array([1], dtype="int64"), name="p2o.helper.constant.36"),
        ),
        make_node(
            "Slice",
            inputs=["p2o.Shape.3", "p2o.helper.constant.35", "p2o.helper.constant.36", "p2o.helper.constant.34"],
            outputs=["p2o.helper.slice.0"],
            name="p2o.Slice.7",
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.37"],
            value=numpy_helper.from_array(np.array([0], dtype="int64"), name="p2o.helper.constant.37"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.38"],
            value=numpy_helper.from_array(np.array([2], dtype="int64"), name="p2o.helper.constant.38"),
        ),
        make_node(
            "Constant",
            inputs=[],
            outputs=["p2o.helper.constant.39"],
            value=numpy_helper.from_array(np.array([4], dtype="int64"), name="p2o.helper.constant.39"),
        ),
        make_node(
            "Slice",
            inputs=["p2o.Shape.3", "p2o.helper.constant.38", "p2o.helper.constant.39", "p2o.helper.constant.37"],
            outputs=["p2o.helper.slice.1"],
            name="p2o.Slice.8",
        ),
        make_node(
            "Concat",
            inputs=["p2o.helper.slice.0", "p2o.helper.constant.33", "p2o.helper.slice.1"],
            outputs=["p2o.helper.concat.0"],
            name="p2o.Concat.2",
            axis=0,
        ),
        make_node(
            "Tile", inputs=["p2o.helper.reshape.0", "p2o.helper.concat.0"], outputs=["p2o.Tile.1"], name="p2o.Tile.0"
        ),
        make_node(
            "ScatterElements",
            inputs=["tmp_5", "p2o.Tile.1", "p2o.Expand.1"],
            outputs=["p2o.tmp_5.0"],
            name="p2o.ScatterElements.0",
            axis=1,
        ),
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
                "p2o.tmp_5.0",
                TensorProto.FLOAT,
                shape=["p2o.DynamicDimension.3", 7, "p2o.DynamicDimension.4", "p2o.DynamicDimension.5"],
            ),
        ]
    )

    save_name = basename + "_pse_post_new" + ext
    save_path = os.path.join(result_path, save_name)
    onnx.save(model, save_path)

    print(f"The new model saved in {save_path}")
