# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Auto mixed precision."""
from __future__ import absolute_import

from mindspore import nn
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.cell_wrapper import _TrainPipelineAccuStepCell
from mindspore.nn.wrap.loss_scale import _TrainPipelineWithLossScaleCell
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_pipeline_stages
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, LossScaleManager
from mindspore import boost, context


AMP_WHITE_LIST = (
    nn.Dense,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Conv1dTranspose,
    nn.Conv2dTranspose,
    nn.Conv3dTranspose
)

AMP_BLACK_LIST = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm
)


class _OutputTo16(nn.Cell):
    """Wrap cell for amp. Cast network output back to float16."""

    def __init__(self, op):
        super(_OutputTo16, self).__init__(auto_prefix=False)
        self._op = op

    def construct(self, x):
        return F.cast(self._op(x), mstype.float16)


class _OutputTo32(nn.Cell):
    "Wrap loss for amp. Cast network output back to float32"

    def __init__(self, backbone):
        super(_OutputTo32, self).__init__(auto_prefix=False)
        self._backbone = backbone

    def construct(self, *inputs):
        out = self._backbone(*inputs)
        return F.mixed_precision_cast(mstype.float32, out)


def _auto_white_list(network, white_list=None):
    """process the white list of network."""
    if white_list is None:
        white_list = AMP_WHITE_LIST
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, white_list):
            network._cells[name] = _OutputTo32(subcell.to_float(mstype.float16))
            change = True
        else:
            _auto_white_list(subcell, white_list)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def _auto_black_list(network, black_list=None):
    """process the black list of network."""
    if black_list is None:
        black_list = AMP_BLACK_LIST
    network.to_float(mstype.float16)
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, black_list):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            _auto_black_list(subcell, black_list)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


def auto_mixed_precision(network, amp_level="O0"):
    """
    auto mixed precision function.

    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: "O0".

            - "O0": Do not change.
            - "O1": [DEMO] Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep operators in black_list run in float32,
            - "O3": Cast network to float16.

    Raises:
        ValueError: If amp level is not supported.

    Examples:
        >>> from mindpsore import amp, nn
        >>> network = LeNet5()
        >>> amp_level = "O1"
        >>> net = amp.auto_mixed_precision(network, amp_level)
    """
    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")
    if amp_level == "O0":
        pass
    elif amp_level == "O1":
        _auto_white_list(network)
        return network
    elif amp_level == "O2":
        _auto_black_list(network)
    elif amp_level == "O3":
        network.to_float(mstype.float16)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))

    if amp_level in ("O2", "O3"):
        network = _OutputTo32(network)
    return network


def _do_keep_batchnorm_fp32(network):
    """Do keep batchnorm fp32."""
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, AMP_BLACK_LIST):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            _do_keep_batchnorm_fp32(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


_config_level = {
    "O0": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float32,
        "loss_scale_manager": None},
    "O1": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float32,
        "loss_scale_manager": None},
    "O2": {
        "keep_batchnorm_fp32": True,
        "cast_model_type": mstype.float16,
        "loss_scale_manager": DynamicLossScaleManager()},
    "O3": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float16,
        "loss_scale_manager": None}}


def _check_kwargs(key_words):
    """Check kwargs."""
    for arg in key_words:
        if arg not in ['cast_model_type', 'keep_batchnorm_fp32', 'loss_scale_manager']:
            raise ValueError(f"Unsupported arg '{arg}'")

    if 'cast_model_type' in key_words:
        validator.check_type_name('cast_model_type', key_words['cast_model_type'],
                                  [mstype.float16, mstype.float32], None)
    if 'keep_batchnorm_fp32' in key_words:
        validator.check_value_type('keep_batchnorm_fp32', key_words['keep_batchnorm_fp32'], bool)
    if 'loss_scale_manager' in key_words:
        loss_scale_manager = key_words['loss_scale_manager']
        if loss_scale_manager:
            validator.check_value_type('loss_scale_manager', loss_scale_manager,
                                       [LossScaleManager, boost.GroupLossScaleManager])


def _check_level(level, boost_level):
    """Check level."""
    if not isinstance(level, str):
        raise TypeError("The argument `level` must be a string in ['O0', 'O1', 'O2', 'O3', 'auto'], \
                         but got type {}.".format(type(level)))
    validator.check('level', level, "", ['O0', 'O1', 'O2', 'O3', 'auto'], Rel.IN)
    validator.check('boost_level', boost_level, "", ['O0', 'O1', 'O2'], Rel.IN)

    if level == "auto":
        device_target = context.get_context('device_target')
        if device_target == "GPU":
            level = "O2"
        elif device_target == "Ascend":
            level = "O3"
        else:
            raise ValueError("Level `auto` only support when `device_target` is GPU or Ascend.")

    enable_boost = False
    if boost_level in ["O1", "O2"]:
        enable_boost = True

    return level, enable_boost


def _add_loss_network(network, loss_fn, cast_model_type):
    """Add loss network."""

    class WithLossCell(nn.Cell):
        "Wrap loss for amp. Cast network output back to float32"

        def __init__(self, backbone, loss_fn):
            super(WithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn

        def construct(self, data, label):
            out = self._backbone(data)
            label = F.mixed_precision_cast(mstype.float32, label)
            return self._loss_fn(F.mixed_precision_cast(mstype.float32, out), label)

    validator.check_value_type('loss_fn', loss_fn, nn.Cell)
    if cast_model_type == mstype.float16:
        network = WithLossCell(network, loss_fn)
    else:
        network = nn.WithLossCell(network, loss_fn)
    return network


def build_train_network(network, optimizer, loss_fn=None, level='O0', boost_level='O0', **kwargs):
    """
    Build the mixed precision training cell automatically.

    Args:
        network (Cell): Definition of the network.
        loss_fn (Union[None, Cell]): Define the loss function. If None, the `network` should have the loss inside.
            Default: None.
        optimizer (Optimizer): Define the optimizer to update the Parameter.
        level (str): Supports ["O0", "O1", "O2", "O3", "auto"]. Default: "O0".

            - "O0": Do not change.
            - "O1": [DEMO] Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O2": Cast network to float16, keep batchnorm and `loss_fn` (if set) run in float32,
              using dynamic loss scale.
            - "O3": Cast network to float16, with additional property `keep_batchnorm_fp32=False` .
            - auto: Set to level to recommended level in different devices. Set level to "O2" on GPU, Set
              level to "O3" Ascend. The recommended level is chosen by the export experience, not applicable to all
              scenarios. User should specify the level for special network.

            "O2" is recommended on GPU, "O3" is recommended on Ascend. Property of `keep_batchnorm_fp32`,
            `cast_model_type` and `loss_scale_manager` determined by `level` setting may be overwritten by settings in
            `kwargs`.

        boost_level (str): Option for argument `level` in `mindspore.boost` , level for boost mode
            training. Supports ["O0", "O1", "O2"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Enable the boost mode, the performance is improved by about 20%, and
              the accuracy is the same as the original accuracy.
            - "O2": Enable the boost mode, the performance is improved by about 30%, and
              the accuracy is reduced by less than 3%.

            If "O1" or "O2" mode is set, the boost related library will take effect automatically.

        cast_model_type (:class:`mindspore.dtype`): Supports `mstype.float16` or `mstype.float32` . If set, the
            network will be casted to `cast_model_type` ( `mstype.float16` or `mstype.float32` ), but not to be casted
            to the type determined by `level` setting.
        keep_batchnorm_fp32 (bool): Keep Batchnorm run in `float32` when the network is set to cast to `float16` . If
            set, the `level` setting will take no effect on this property.
        loss_scale_manager (Union[None, LossScaleManager]): If not None, must be subclass of
            :class:`mindspore.amp.LossScaleManager` for scaling the loss. If set, the `level` setting will
            take no effect on this property.

    Raises:
        ValueError: If device is CPU, property `loss_scale_manager` is not `None` or `FixedLossScaleManager`
            (with property `drop_overflow_update=False` ).

    Examples:
        >>> from mindpsore import amp, nn
        >>> network = LeNet5()
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
        >>> net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
        >>> amp_level="O3"
        >>> net = amp.build_train_network(network, net_opt, net_loss, amp_level)
    """
    validator.check_value_type('optimizer', optimizer, (nn.Optimizer, boost.FreezeOpt,
                                                        nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell))

    level, enable_boost = _check_level(level, boost_level)

    _check_kwargs(kwargs)
    config = dict(_config_level.get(level), **kwargs)

    if config["cast_model_type"] == mstype.float16:
        network.to_float(mstype.float16)

        if config["keep_batchnorm_fp32"]:
            _do_keep_batchnorm_fp32(network)
    elif not config["keep_batchnorm_fp32"] and level == "O2":
        network.to_float(mstype.float16)
    elif config["cast_model_type"] == mstype.float32 and level in ("O2", "O3"):
        pass
    else:
        auto_mixed_precision(network, level)

    if loss_fn:
        network = _add_loss_network(network, loss_fn, config["cast_model_type"])

    loss_scale = 1.0
    if config["loss_scale_manager"] is not None:
        loss_scale_manager = config["loss_scale_manager"]
        loss_scale = loss_scale_manager.get_loss_scale()
        update_cell = loss_scale_manager.get_update_cell()
        if update_cell is not None:
            # only cpu not support `TrainOneStepWithLossScaleCell` for control flow.
            if not context.get_context("enable_ge") and context.get_context("device_target") == "CPU":
                raise ValueError("Only `loss_scale_manager=None` or "
                                 "`loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False)`"
                                 "are supported on device `CPU`. ")
            if _get_pipeline_stages() > 1:
                network = _TrainPipelineWithLossScaleCell(network, optimizer,
                                                          scale_sense=update_cell).set_train()
            elif enable_boost:
                network = boost.BoostTrainOneStepWithLossScaleCell(network, optimizer,
                                                                   scale_sense=update_cell).set_train()
            else:
                network = nn.TrainOneStepWithLossScaleCell(network, optimizer,
                                                           scale_sense=update_cell).set_train()
            return network
    if _get_pipeline_stages() > 1:
        network = _TrainPipelineAccuStepCell(network, optimizer).set_train()
    elif enable_boost:
        network = boost.BoostTrainOneStepCell(network, optimizer, loss_scale).set_train()
    else:
        network = nn.TrainOneStepCell(network, optimizer, loss_scale).set_train()
    return network
