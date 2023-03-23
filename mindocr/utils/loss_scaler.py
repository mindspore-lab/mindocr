from mindspore import nn


def get_loss_scales(cfg):
    '''
    Args:
        cfg (dict): configure dict of loss scaler
    
    Returns:
        nn.Cell: scale_sens used to scale gradient    
        float: loss_scale used in optimizer (only used when loss scaler type is static and drop_overflow update is False) 
    '''
    # loss scale is 1.0 by default
    loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale_value=1.0)
    optimizer_loss_scale = 1.0 
    
    if 'loss_scaler' in cfg: 
        assert 'loss_scale' in cfg.loss_scaler, 'Must specify the value for `loss_scale` in the config if `loss_scaler` is used.'
        if cfg.loss_scaler.type == 'dynamic':
            # TODO: scale_window can be related to num_batches, e.g., scale_window = num_batches * 2
            loss_scale_manager = nn.DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scaler.get('loss_scale', 2**16), 
                                                            scale_factor=cfg.loss_scaler.get('scale_factor', 2.0), 
                                                            scale_window=cfg.loss_scaler.get('scale_window', 2000), 
                                                            )
        elif cfg.loss_scaler.type == 'static':
            loss_scale = cfg.loss_scaler.get('loss_scale', 1.0)
            loss_scale_manager = nn.FixedLossScaleUpdateCell(loss_scale)
            # when using static loss scaler and drop_overflow_update is False, we should also set loss_scale for optimizer.
            if not cfg.system.drop_overflow_update: 
                optimizer_loss_scale = loss_scale 
        else:
            raise ValueError(f'Available loss scaler types are `static` and `dynamic`, but got {cfg.loss_scaler}')
    
    return loss_scale_manager, optimizer_loss_scale 

