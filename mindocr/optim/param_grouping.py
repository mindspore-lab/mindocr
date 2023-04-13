'''
group parameters for setting different weight decay or learning rate for different layers in the network.
'''
__all__ = ['build_group_params']

supported_grouping_strategies = ['svtr', 'filter_norm_and_bias']

def grouping_default(params, weight_decay):
    decay_params = []
    no_decay_params = []

    filter_keys = ['beta', 'gamma', 'bias']
    for param in params:
        filter_param = False
        for k in filter_keys:
            if k in param.name:
                filter_param = True

        if filter_param:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def grouping_svtr(params, weight_decay):
    decay_params = []
    no_decay_params = []

    filter_keys = ['beta', 'gamma', 'pos_emb', 'bias'] # correspond to nn.BatchNorm, nn.LayerNorm, and position embedding layer if named as 'pos_emb'. TODO: check svtr naming.

    for param in params:
        filter_param = False
        for k in filter_keys:
            if k in param.name:
                filter_param = True

        if filter_param:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params},
        {"order_params": params},
    ]


def create_group_params(params, weight_decay=0, grouping_strategy=None, no_weight_decay_params=[], **kwargs):
    '''
    create group parameters for setting different weight decay or learning rate for different layers in the network.

    Args:
        params: network params
        weight_decay (float): weight decay value
        group_strategy (str): name of the hard-coded grouping strategy. If not None, group parameters according to the predefined function and `no_weight_decay_params` will not make effect.
        no_weight_decay_params (list): list of the param name substrings that will be picked to exclude from weight decay. If a parameter containing one of the substrings in the list, the paramter will not be applied with weigt decay. (Tips: param names can be checked by `[p.name for p in network.trainable_params()]`

    Return:
        list[dict], grouped parameters
    '''
    gp = grouping_strategy
    if gp is not None:
        if weight_decay == 0:
            print("WARNING: weight decay is 0 in param grouping.")
        if gp == 'svtr':
            return grouping_svtr(params, weight_decay)
        elif gp == 'filter_norm_and_bias':
            return grouping_default(params, weight_decay)
        else:
            raise ValueError(f'The grouping function for {gp} is not defined. Valid grouping strategies are {supported_grouping_strategies}')

    elif len(no_weight_decay_params) > 0:
        assert weight_decay > 0, f'Invalid weight decay value {weight_decay} for param grouping.'
        decay_params = []
        no_decay_params = []
        for param in params:
            filter_param = False
            for k in no_weight_decay_params:
                if k in param.name:
                    filter_param = True

            if filter_param:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params},
            {"order_params": params},
        ]
    else:
        print("INFO: no parameter grouping is applied.")
        return params


