import logging

from mindspore import nn

_logger = logging.getLogger(__name__)


def get_loss_scales(cfg):
    """
    Args:
        cfg (dict): configure dict of loss scaler

    Returns:
        nn.Cell: scale_sens used to scale gradient
        float: loss_scale used in optimizer
            (only used when loss scaler type is static and drop_overflow update is False)
    """
    # loss scale is 1.0 by default
    loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale_value=1.0)

    # Only when `FixedLossScaleManager` is used for training and the `drop_overflow_update` in
    # `FixedLossScaleManager` is set to False, then this value needs to be the same as the `loss_scale` in
    # `FixedLossScaleManager`
    # But we never use FixedLossScaleManager, so optimizer_loss_scale is always 1.
    optimizer_loss_scale = 1.0

    if "loss_scaler" in cfg:
        assert (
            "loss_scale" in cfg.loss_scaler
        ), "Must specify the value for `loss_scale` in the config if `loss_scaler` is used."
        if cfg.loss_scaler.type == "dynamic":
            # TODO: scale_window can be related to num_batches, e.g., scale_window = num_batches * 2
            scale_factor = cfg.loss_scaler.get("scale_factor", 2.0)
            scale_window = cfg.loss_scaler.get("scale_window", 2000)
            # adjust by gradient_accumulation_steps so that the scaling process is the same as that of
            # batch_size=batch_size*gradient_accumulation_steps
            grad_accu_steps = cfg.train.get("gradient_accumulation_steps", 1)
            if grad_accu_steps > 1:
                scale_factor = scale_factor ** (1 / grad_accu_steps)
                scale_window = scale_window * grad_accu_steps
                _logger.info(
                    "gradient_accumulation_steps > 1, scale_factor and scale_window are adjusted accordingly for "
                    "dynamic loss scaler"
                )
            loss_scale_manager = nn.DynamicLossScaleUpdateCell(
                loss_scale_value=cfg.loss_scaler.get("loss_scale", 2**16),
                scale_factor=scale_factor,
                scale_window=scale_window,
            )
        elif cfg.loss_scaler.type == "static":
            loss_scale = cfg.loss_scaler.get("loss_scale", 1.0)
            loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale)
        else:
            raise ValueError(f"Available loss scaler types are `static` and `dynamic`, but got {cfg.loss_scaler}")

    return loss_scale_manager, optimizer_loss_scale
