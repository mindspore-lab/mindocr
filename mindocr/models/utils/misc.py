from packaging import version

import mindspore as ms
from mindspore.ops.primitive import constexpr


@constexpr
def ms_version_is_large_than_2_0():
    """This check can be applied in `nn.Cell.construct` method, to
    make compatibilities in differenct Mindspore version
    """
    return version.parse(ms.__version__) >= version.parse("2.0.0rc")
