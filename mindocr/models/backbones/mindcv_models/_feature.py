from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Tuple

import mindspore.nn as nn
from mindspore import Tensor


def _cell_list(net: nn.Cell, flatten_sequential: bool = False) -> Iterable[Tuple[str, str, nn.Cell]]:
    """Yield the partially flattened cell list from the model, together with its new name and old name

    Args:
        net (nn.Cell): Network need to be partially flattened
        flatten_sequential (bool): Flatten the inner-layer of the sequential cell. Default: False.

    Returns:
        iterator[tuple[str, str, nn.Cell]]: The new name, the old name and corresponding cell
    """
    for name, cell in net.name_cells().items():
        if flatten_sequential and isinstance(cell, nn.SequentialCell):
            for child_name, child_cell in cell.name_cells().items():
                combined = [name, child_name]
                yield "_".join(combined), ".".join(combined), child_cell
        else:
            yield name, name, cell


def _get_return_layers(feature_info: Dict[str, Any], out_indices: List[int]) -> Dict[str, int]:
    """Create a dict storing the "layer_name - layer_id" pair that need to be extracted"""
    return_layers = dict()
    for i, x in enumerate(feature_info):
        if i in out_indices:
            return_layers[x["name"]] = i
    return return_layers


class FeatureExtractWrapper(nn.Cell):
    """A wrapper of the original model, aims to perform the feature extraction at each stride.
    Basically, it performs 3 steps: 1. extract the return node name from the network's property
    `feature_info`; 2. partially flatten the network architecture if network's attribute `flatten_sequential`
    is True; 3. rebuild the forward steps and output the features based on the return node name.

    It also provide a property `out_channels` in the wrapped model, return the number of features at each output
    layer. This propery is usually used for the downstream tasks, which requires feature infomation at network
    build stage.

    It should be note that to apply this wrapper, there is a strong assumption that each of the outmost cell
    are registered in the same order as they are used. And there should be no reuse of each cell, even for the `ReLU`
    cell. Otherwise, the returned result may not be correct.

    And it should be also note that it basically rebuild the model. So the default checkpoint parameter cannot be loaded
    correctly once that model is wrapped. To use the pretrained weight, please load the weight first and then use this
    wrapper to rebuild the model.

    Args:
        net (nn.Cell): The model need to be wrapped.
        out_indices (list[int]): The indicies of the output features. Default: [0, 1, 2, 3, 4]
    """

    def __init__(self, net: nn.Cell, out_indices: List[int] = [0, 1, 2, 3, 4]) -> None:
        super().__init__(auto_prefix=False)

        feature_info = self._get_feature_info(net)
        flatten_sequetial = getattr(net, "flatten_sequential", False)
        cells = _cell_list(net, flatten_sequential=flatten_sequetial)
        return_layers = _get_return_layers(feature_info, out_indices)
        self.net, updated_return_layers = self._create_net(cells, return_layers)

        # calculate the return index
        self.return_index = list()
        for i, name in enumerate(self.net.name_cells().keys()):
            if name in updated_return_layers:
                self.return_index.append(i)

        # calculate the out_channels
        self._out_channels = list()
        for i in return_layers.values():
            self._out_channels.append(feature_info[i]["chs"])

    @property
    def out_channels(self):
        """The output channels of the model, filtered by the out_indices.
        """
        return self._out_channels

    def construct(self, x: Tensor) -> List[Tensor]:
        return self._collect(x)

    def _get_feature_info(self, net: nn.Cell) -> Dict[str, Any]:
        try:
            feature_info = getattr(net, "feature_info")
        except AttributeError:
            raise
        return feature_info

    def _create_net(
        self, cells: Iterable[Tuple[str, str, nn.Cell]], return_layers: Dict[str, int]
    ) -> Tuple[nn.SequentialCell, Dict[str, int]]:
        layers = OrderedDict()
        updated_return_layers = dict()
        remaining = set(return_layers.keys())
        for new_name, old_name, module in cells:
            layers[new_name] = module
            if old_name in remaining:
                updated_return_layers[new_name] = return_layers[old_name]
                remaining.remove(old_name)
            if not remaining:
                break

        net = nn.SequentialCell(layers)
        return net, updated_return_layers

    def _collect(self, x: Tensor) -> List[Tensor]:
        out = list()
        for i, cell in enumerate(self.net.cell_list):
            x = cell(x)
            if i in self.return_index:
                out.append(x)
        return out
