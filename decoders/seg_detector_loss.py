import sys
from .loss.balance_cross_entropy_loss import BalanceCrossEntropyLoss

import torch
import torch.nn as nn


class SegDetectorLossBuilder():
    '''
    Build loss functions for SegDetector.
    Details about the built functions:
        Input:
            pred: A dict which contains predictions.
                thresh: The threshold prediction
                binary: The text segmentation prediction.
                thresh_binary: Value produced by `step_function(binary - thresh)`.
            batch:
                gt: Text regions bitmap gt.
                mask: Ignore mask,
                    pexels where value is 1 indicates no contribution to loss.
                thresh_mask: Mask indicates regions cared by thresh supervision.
                thresh_map: Threshold gt.
        Return:
            (loss, metrics).
            loss: A scalar loss value.
            metrics: A dict contraining partial loss values.
    '''

    def __init__(self, loss_class, *args, **kwargs):
        self.loss_class = loss_class
        self.loss_args = args
        self.loss_kwargs = kwargs

    def build(self):
        return getattr(sys.modules[__name__], self.loss_class)(*self.loss_args, **self.loss_kwargs)

class sizemseloss(nn.Module):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=1, bce_scale=6):
        super(sizemseloss, self).__init__()

        from .loss.balance_cross_entropy_loss import BalanceCrossEntropyLoss
        self.mse_loss = torch.nn.MSELoss()
        self.bce_loss = BalanceCrossEntropyLoss()
        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):

        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])

        size_loss = self.mse_loss(pred['size'][:,0,:,:]*batch['mask'], batch['size']*batch['mask'])

        loss =  bce_loss * 6 + self.l1_scale * size_loss
        metrics = dict(bce_loss=bce_loss)
        metrics['size_loss'] = size_loss
        return loss, metrics 
