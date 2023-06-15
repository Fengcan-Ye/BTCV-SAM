import torch

n_categories = 13


def single_point_prompt(labels : torch.Tensor, device=None):
    """
    参数：
    labels: torch.Tensor[B, H, W]

    返回：
    input_points: [dict of organid : torch.Tensor[2, ]]
    """

    batchsize = labels.shape[0]
    if device is None:
        device = labels.device 
    
    input_points = [dict() for _ in range(batchsize)]

    for organ_id in range(1, n_categories + 1):
        # for every organ 
        organ_mask = labels == organ_id 

        for i in range(batchsize):
            slice = organ_mask[i]

            if torch.any(slice):
                segmentation = torch.where(slice) # tuple[tensor, tensor]
                n_pixels = len(segmentation[0])
                selected = torch.randint(0, n_pixels, (1,))

                input_points[i][organ_id] = torch.tensor([segmentation[1][selected], segmentation[0][selected]], device=device)
    
    return input_points



def multi_point_prompt(labels : torch.Tensor, device=None):
    raise NotImplementedError

def box_prompt(labels : torch.Tensor, device=None):
    """
    参数：
    labels: torch.Tensor[B, H, W]
    device: bounding box所在的设备, 默认与labels一致

    返回：
    input_boxes: [dict of organ_id : bbox]
    """
    batchsize = labels.shape[0]
    if device is None:
        device = labels.device
    input_boxes = [dict() for _ in range(batchsize)]

    for organ_id in range(1, n_categories + 1):
        # for every organ 
        organ_mask = labels == organ_id

        for i in range(batchsize):
            slice = organ_mask[i]

            if torch.any(slice):
                segmentation = torch.where(slice)

                xmin = torch.min(segmentation[1])
                xmax = torch.max(segmentation[1])
                ymin = torch.min(segmentation[0])
                ymax = torch.max(segmentation[0])

                input_boxes[i][organ_id] = torch.tensor([xmin, ymin, xmax, ymax], device=device)
    
    return input_boxes