from mindspore import nn
import mindspore.ops as ops
from mindspore.communication import get_group_size


class NetWithLossWrapper(nn.Cell):
    '''
    A universal wrapper for any network with any loss.

    Assume dataloader output follows the order (input1, input2, ..., label1, label2, label3, ... )  for network and loss.

    Args:
        net (nn.Cell): network
        loss_fn: loss function
        num_net_inputs: number of network input, e.g. 1
        num_labels: number of labels used for loss fn computation. If None, all the remaining args will be fed into loss func.
    '''
    def __init__(self, net, loss_fn, num_net_inputs=1, num_labels=None):
        super().__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn
        # TODO: get this automatically from net and loss func
        self.num_net_inputs = num_net_inputs
        self.num_labels = num_labels
        #self.net_forward_input = ['img']
        #self.loss_forward_input = ['gt', 'gt_mask', 'thresh_map', 'thresh_mask']

    def construct(self, *args):
        '''
        Args:
            args (Tuple): contains network inputs, labels (given by data loader)
        Returns:
            loss_val (Tensor): loss value
        '''
        pred = self._net(*args[:self.num_net_inputs])
        if self.num_labels is None:
            loss_val = self._loss_fn(pred, *args[self.num_net_inputs:])
        else:
            loss_val = self._loss_fn(pred, *args[self.num_net_inputs:self.num_net_inputs+self.num_labels])

        return loss_val

class NetWithEvalWrapper(nn.Cell):
    '''
    A universal wrapper for any network with any loss for evaluation pipeline.
    Difference from NetWithLossWrapper: it returns loss_val, pred, and labels.

    Assume dataloader output follows the order (input1, input2, ..., label1, label2, label3, ... )  for network and loss.

    Args:
        net (nn.Cell): network
        loss_fn: loss function, if None, will not compute loss for evaluation dataset
        num_net_inputs: number of network input, e.g. 1
        num_labels: number of labels used for loss fn computation. If None, all the remaining args will be fed into loss func.
    '''
    def __init__(self, net, loss_fn=None, num_net_inputs=1, num_labels=None):
        super().__init__(auto_prefix=False)
        self._net = net
        self._loss_fn = loss_fn
        # TODO: get this automatically from net and loss func
        self.num_net_inputs = num_net_inputs
        self.num_labels = num_labels
        #self.net_forward_input = ['img']
        #self.loss_forward_input = ['gt', 'gt_mask', 'thresh_map', 'thresh_mask']

    def construct(self, *args):
        '''
        Args:
            args (Tuple): contains network inputs, labels (given by data loader)
        Returns:
            Tuple: loss value (Tensor), pred (Union[Tensor, Tuple[Tensor]]), labels (Tuple)
        '''
        # TODO: pred is a dict
        pred = self._net(*args[:self.num_net_inputs])
        if self.num_labels is None:
            labels = args[self.num_net_inputs:]
        else:
            labels = args[self.num_net_inputs:self.num_net_inputs+self.num_labels]

        if self._loss_fn is not None:
            loss_val = self._loss_fn(pred, *labels)
        else:
            loss_val = None

        return loss_val, pred, labels


class DBNetWithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        net (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, net, loss_fn):
        super().__init__(auto_prefix=False)

        self._net = net
        self._loss_fn = loss_fn

    # Note: this order should be consistent with the dataloader output
    def construct(self, img, gt, gt_mask, thresh_map, thresh_mask):
        pred = self._net(img)
        loss = self._loss_fn(pred, gt, gt_mask, thresh_map, thresh_mask)

        return loss

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._network
