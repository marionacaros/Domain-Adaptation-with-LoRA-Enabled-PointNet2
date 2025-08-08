import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification tasks with class imbalance.
    It down-weights well-classified examples and focuses training on hard examples.

    Args:
        gamma (float): Focusing parameter. Higher gamma -> more focus on hard examples.
        weight (Tensor, optional): Per-class weighting tensor.
        ignore_index (int): Class to ignore in loss computation.
    """
    def __init__(self, gamma=2.0, weight=None, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logpt, targets):
        
        # logpt are log-probabilities  # [B, N, C]
        # targets [B, N]

        # Apply log_softmax to get log-probabilities
        # logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.permute(0, 2, 1) # [B, C, N]
        pt = torch.exp(logpt)  

        # Gather log-probability and probability for the correct class
        # We need to use gather along dim=1 (class dim), so we prepare index accordingly:
        targets_expanded = targets.unsqueeze(1)  # [B, 1, N]

        logpt = logpt.gather(1, targets_expanded).squeeze(1)  # [B, N]
        pt = pt.gather(1, targets_expanded).squeeze(1)        # [B, N]

        # Compute the focal loss
        loss = -((1 - pt) ** self.gamma) * logpt  # [B, N]

        # Mask out ignore_index if provided
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
            loss = loss * mask

        # Apply per-class weights
        if self.weight is not None:
            w = self.weight[targets]  # [B, N]
            loss = loss * w

        return loss.mean()



class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks, useful for handling class imbalance
    by directly optimizing overlap between prediction and ground truth.

    Args:
        smooth (float): Small constant to avoid division by zero.
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs [B,N,C]
        # Apply softmax to get class probabilities
        # inputs = F.softmax(inputs, dim=1)  # [B, C, N]

        # One-hot encode the targets for dice computation
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[2])  # [B, N, C]
        # targets_one_hot = targets_one_hot.permute(0, 2, 1).float()         # [B, C, N]

        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=2)  # [B, C]
        union = inputs.sum(dim=2) + targets_one_hot.sum(dim=2)  # [B, C]

        # Compute Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return 1 - mean Dice (i.e., Dice Loss)
        loss = 1 - dice.mean()
        return loss

    

class CombinedLoss(nn.Module):
    """
    Combines Focal Loss and Dice Loss to balance classification confidence and spatial overlap.

    Args:
        dice_weight (float): Weight for the Dice Loss component.
        focal_weight (float): Weight for the Focal Loss component.
    """
    def __init__(self, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        # Weighted sum of Focal Loss and Dice Loss
        focal = self.focal(inputs, targets)
        dice = self.dice(inputs, targets)
        return self.focal_weight * focal + self.dice_weight * dice


class CELoss(nn.Module):
    
    def __init__(self, weight, reduction='mean'):
        super(CELoss, self).__init__()
        self.celoss = torch.nn.CrossEntropyLoss(weight, reduction)

    def forward(self, logpt, targets):

        logpt = logpt.contiguous().view(-1, logpt.shape[-1]) # [batch* 4096, C]
        targets = targets.contiguous().view(-1) # [batch* 4096]

        loss = self.celoss(logpt,targets)

        return loss
