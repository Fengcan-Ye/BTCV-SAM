import torch 

def dice(pred_mask, gt_mask):
    """
      参数：
      pred_mask & gt_mask: [B, H, W] contains only True/False values

      返回值：
      dice: torch.Tensor [B]
    """ 
    intersect = torch.sum(torch.logical_and(pred_mask, gt_mask), dim=(1, 2))
    return 2 * intersect / (torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)))