import torch.nn as nn
import torch
from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np
cfs = ConfusionMatrix(3)
class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1-dice
    torch.nn.BCEWithLogitsLoss

class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
                     
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU
