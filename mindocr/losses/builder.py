import inspect
from .det_loss import L1BalancedCELoss
from .rec_loss import CTCLoss 


support_losses = ['L1BalancedCELoss', 'CTCLoss']

def build_loss(name, **kwargs):
    '''
    Args:
        name: loss name, exactly the same as one of the supported loss class names
        
    '''
    assert name in support_losses, f'Invalid loss name {name}, support losses are {support_losses}'

    loss_fn = eval(name)(**kwargs)

    #print('loss func inputs: ', loss_fn.construct.__code__.co_varnames)
    print('==> Loss func input args: \n\t', inspect.signature(loss_fn.construct))

    return loss_fn
        
    
    
