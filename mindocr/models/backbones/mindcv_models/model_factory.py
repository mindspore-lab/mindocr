import os

from mindspore import load_checkpoint, load_param_into_net

from .registry import is_model, model_entrypoint

__all__ = ["create_model"]


def create_model(
    model_name: str,
    num_classes: int = 1000,
    pretrained=False,
    in_channels: int = 3,
    checkpoint_path: str = "",
    ema=False,
    **kwargs,
):
    r"""Creates model by name.

    Args:
        model_name (str):  The name of model.
        num_classes (int): The number of classes. Default: 1000.
        pretrained (bool): Whether to load the pretrained model. Default: False.
        in_channels (int): The input channels. Default: 3.
        checkpoint_path (str): The path of checkpoint files. Default: "".
        ema (bool): Whether use ema method. Default: False.
    """

    if checkpoint_path != "" and pretrained:
        raise ValueError("checkpoint_path is mutually exclusive with pretrained")

    model_args = dict(num_classes=num_classes, pretrained=pretrained, in_channels=in_channels)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not is_model(model_name):
        raise RuntimeError(f"Unknown model {model_name}")

    create_fn = model_entrypoint(model_name)
    model = create_fn(**model_args, **kwargs)

    if os.path.exists(checkpoint_path):
        checkpoint_param = load_checkpoint(checkpoint_path)
        ema_param_dict = dict()
        for param in checkpoint_param:
            if param.startswith("ema"):
                new_name = param.split("ema.")[1]
                ema_data = checkpoint_param[param]
                ema_data.name = new_name
                ema_param_dict[new_name] = ema_data

        if ema_param_dict and ema:
            load_param_into_net(model, ema_param_dict)
        elif bool(ema_param_dict) is False and ema:
            raise ValueError("chekpoint_param does not contain ema_parameter, please set ema is False.")
        else:
            load_param_into_net(model, checkpoint_param)

    return model
