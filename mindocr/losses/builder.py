import inspect
from .det_loss import L1BalancedCELoss
from .rec_loss import CTCLoss 


supported_losses = ['L1BalanceCELoss', 'CTCLoss']


def build_loss(name, **kwargs):
    '''
    Args:
        name: loss name, exactly the same as one of the supported loss class names
        
    '''
    assert name in supported_losses, f'Invalid loss name {name}, support losses are {supported_losses}'

    loss_fn = eval(name)(**kwargs)

    #print('loss func inputs: ', loss_fn.construct.__code__.co_varnames)
    print('==> Loss func input args: \n\t', inspect.signature(loss_fn.construct))

    return loss_fn