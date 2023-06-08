from mindspore.common.initializer import Constant, initializer


def initialize_network_with_constant(network, c_weight=0.5, c_bias=0.0):
    for name, param in network.parameters_and_names():
        if "weight" in name:
            param.set_data(initializer(Constant(c_weight), param.shape, param.dtype))
        elif "bias" in name:
            param.set_data(initializer(Constant(c_bias), param.shape, param.dtype))
        else:
            param.set_data(initializer(Constant(0.0), param.shape, param.dtype))
    """
    if isinstance(cell, nn.BatchNorm2d):
        cell.beta.set_data(initializer(Constant(0), cell.beta.shape, cell.beta.dtype))
        cell.gamma.set_data(initializer(Constant(1), cell.gamma.shape, cell.gamma.dtype))
    elif isinstance(cell, nn.Conv2d):
        cell.weight.set_data(initializer(Constant(0.02), cell.weight.shape, cell.weight.dtype))
        if cell.bias is not None:
            cell.bias.set_data(initializer(Constant(0), cell.bias.shape, cell.bias.dtype))
    """
