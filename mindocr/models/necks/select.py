__all__ = ['Select']

class Select(object):
    '''
    select feature from the backbone output.
    '''
    def __init__(self, in_channels, index=-1):
        self.index = index
        self.out_channels = in_channels[index]

    def __call__(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            return x[self.index]
        else:
            return x
