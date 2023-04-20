"""
Create and run transformations from a config or predefined transformation pipeline
"""
from typing import List, Tuple, Callable

# FIXME: make sure our transform names don't overlap with MS
from mindspore.dataset.vision import *
from .general_transforms import *
from .det_transforms import *
from .rec_transforms import *
from .iaa_augment import *

__all__ = ['create_transforms']


def _unfold_dict(transform_func: Callable, input_cols: List[str], op_in_cols: List[str] = None,
                 op_out_cols: List[str] = None) -> Callable:
    def unfold_dict(*args):
        """
        1. Pack the column names and data tuples into a dictionary
        2. Calling the transform method on the dictionary
        3. Unpack the dictionary and return the data tuples only
        """
        input_data = dict(zip(input_cols, args))
        if op_in_cols is not None:  # for MS built-in transform
            transformed = transform_func(*[input_data[name] for name in op_in_cols])
            transformed = dict(zip(op_out_cols, transformed)) if isinstance(transformed, tuple) \
                else {op_out_cols[0]: transformed}
            input_data.update(transformed)
            transformed = input_data    # return full dictionary
        else:                       # for MindOCR transform
            transformed = transform_func(input_data)

        return tuple(transformed.values())

    return unfold_dict


# TODO: use class with __call__, to perform transformation
def create_transforms(transform_pipeline, input_columns: List[str],
                      global_config=None) -> Tuple[List[Callable], List[str]]:
    """
    Create a sequence of callable transforms.

    Args:
        transform_pipeline (List): list of dicts where each key is a transformation class name,
                                   and its value are the args.
            e.g. [{'DecodeImage': {'img_mode': 'BGR', 'channel_first': False}}]
                 [DecodeImage(img_mode='BGR')]
        input_columns: list of input columns to data pipeline
    Returns:
        tuple: list of data transformation callable functions, data pipeline output columns
    """
    assert isinstance(transform_pipeline, list), \
        f'transform_pipeline config should be a list, but {type(transform_pipeline)} detected'

    _transforms, output_columns = [], input_columns
    for transform_config in transform_pipeline:
        assert len(transform_config) == 1, "yaml format error in transforms"
        trans_name = list(transform_config.keys())[0]
        param = {} if transform_config[trans_name] is None else transform_config[trans_name]
        #  TODO: not each transform needs global config
        if global_config is not None:
            param.update(global_config)
        # TODO: assert undefined transform class

        transform = eval(trans_name)

        op_in_cols, op_out_cols = None, None
        if issubclass(transform, transforms.TensorOperation):   # if MS built-in transform
            if 'input_columns' in param:
                op_in_cols = param.pop('input_columns')
                op_out_cols = param.pop('output_columns')
            else:   # transforms performed on image by default
                op_in_cols, op_out_cols = ['image'], ['image']

            transform = transform(**param)  # NOQA
            new_output = op_out_cols

        else:                                                   # for MindOCR transform
            transform = transform(**param)
            new_output = transform.output_columns

        _transforms.append(_unfold_dict(transform, output_columns.copy(), op_in_cols, op_out_cols))
        output_columns.extend([oc for oc in new_output if oc not in output_columns])
    return _transforms, output_columns
