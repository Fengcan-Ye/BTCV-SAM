import torch 
import torch.nn as nn
import torch.nn.functional as F

@torch.no_grad()
def dice(pred_mask, gt_mask):
    """
      参数：
      pred_mask & gt_mask: torch.Tensor[B, H, W] contains only True/False values

      返回值：
      dice: torch.Tensor [B]
    """ 
    intersect = torch.sum(torch.logical_and(pred_mask, gt_mask), dim=(1, 2))
    return 2 * intersect / (torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)))

@torch.no_grad()
def IoU(pred_mask, gt_mask):
    """
      参数：
      pred_mask & gt_mask: torch.Tensor[B, H, W] contains only True/False values

      返回值：
      IoU: torch.Tensor [B]
    """ 
    intersect = torch.sum(torch.logical_and(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(torch.logical_or(pred_mask, gt_mask), dim=(1,2))

    return intersect / union

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth = 1):
        """
        参数：
          inputs: torch.Tensor[B, H, W] predicted logits
          targets: torch.Tensor[B, H, W] ground truth binary masks
          smooth: float
        返回：
          dice loss
        """

        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
    

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
    
    def forward(self, inputs, targets, alpha=0.8, gamma=2):
        """
        参数：
          inputs: torch.Tensor[B, H, W] predicted logits
          targets: torch.Tensor[B, H, W] ground truth binary masks
          alpha: float
          gamma: float
        返回：
          focal loss
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        BCE = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE 

        return focal_loss