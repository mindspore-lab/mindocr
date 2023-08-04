"""
Create and run transformations from a config or predefined transformation pipeline
"""
from typing import Callable, List

import mindspore.dataset.vision as ms_vision

from .det_east_transforms import *
from .det_fce_transforms import *
from .det_transforms import *
from .general_transforms import *

# from .rec_abinet_transforms import *
from .rec_transforms import *
from .svtr_transform import *

__all__ = ["create_transforms"]


def _mindocr_transforms(_transforms: List[Callable], input_cols: List[str], output_cols: List[str]) -> Callable:
    """
    Pack together custom MindOCR transforms into a dictionary, perform transforms, and unpack back to a tuple for
    increased performance with MindSpore data pipeline.
    Args:
        _transforms: list of MindOCR transforms to apply to an input.
        input_cols: input column names for the transforms.
        output_cols: output column names for the transforms.

    Returns:
        callable function that applies custom MindOCR transforms.
    """

    def run_transforms(*data_tuple):
        transformed = dict(zip(input_cols, data_tuple))  # Pack column names and data tuple into a dictionary
        for transform in _transforms:
            transformed = transform(transformed)

        # ensure order of the values to be consistent with output_cols
        assert all(
            [key in transformed for key in output_cols]
        ), f"At least one key in {output_cols} does not exist \
            in the transformed dictionary after {_transforms}."
        return tuple([transformed[key] for key in output_cols])  # Unpack the dictionary and return the data tuple only

    return run_transforms


# TODO: use class with __call__, to perform transformation
def create_transforms(
    transform_pipeline: List[dict], input_columns: List[str], global_config: dict = None, backward_comp=False
) -> List[dict]:
    """
    Create a sequence of callable transforms.

    Args:
        transform_pipeline (List): list of dicts where each key is a transformation class name,
                                   and its value are the args.
            e.g. [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                 [DecodeImage(img_mode='BGR')]
        input_columns: list of input columns to the data pipeline
        backward_comp: Ensure backwards compatibility with MS1.x by providing required `column_order` parameter
    Returns:
        tuple: list of dictionaries with transformations and their parameters
    """
    assert isinstance(
        transform_pipeline, list
    ), f"transform_pipeline config should be a list, but {type(transform_pipeline)} detected"

    _transforms = []
    input_columns, output_columns = input_columns.copy(), input_columns.copy()
    for transform_config in transform_pipeline:
        assert len(transform_config) == 1, "yaml format error in transforms"
        trans_name = list(transform_config.keys())[0]
        param = {} if transform_config[trans_name] is None else transform_config[trans_name]
        if global_config is not None:
            param.update(global_config)

        try:  # check if it is a MindOCR transform first
            transform = eval(trans_name)(**param)
            output_columns.extend([oc for oc in transform.output_columns if oc not in output_columns])

            if _transforms and not _transforms[-1]["MS_operation"]:
                _transforms[-1]["operations"].append(transform)
                _transforms[-1]["output_columns"] = output_columns
            else:
                _transforms.append(
                    {
                        "operations": [transform],
                        "input_columns": input_columns,
                        "output_columns": output_columns.copy(),
                        "MS_operation": False,
                    }
                )

        except NameError:  # if MS built-in transform
            op_in_cols = param.pop("input_columns", ["image"])
            op_out_cols = param.pop("output_columns", ["image"])
            output_columns.extend([oc for oc in op_out_cols if oc not in output_columns])

            transform = getattr(ms_vision, trans_name)(**param)
            if _transforms and _transforms[-1]["MS_operation"]:
                _transforms[-1]["operations"].append(transform)  # NOQA
            else:
                _transforms.append(
                    {
                        "operations": [transform],
                        "input_columns": op_in_cols,
                        "output_columns": op_out_cols,
                        "MS_operation": True,
                    }
                )

        input_columns = output_columns.copy()
        if backward_comp:
            _transforms[-1]["column_order"] = output_columns.copy()

    wrapped_transforms = []
    for transform in _transforms:
        ms_op = transform.pop("MS_operation")
        if not ms_op:
            transform["operations"] = _mindocr_transforms(
                transform["operations"], transform["input_columns"], transform["output_columns"]
            )

        wrapped_transforms.append(transform)

    return wrapped_transforms
