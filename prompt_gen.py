import torch
import random

n_categories = 13


def point_prompt(labels : torch.Tensor, n_points=1, device=None, single_prompt=False):
    """
    参数：
    labels: torch.Tensor[B, H, W]

    返回：
    input_points: [dict of organid : torch.Tensor[n_points, 2]]
    """

    batchsize = labels.shape[0]
    if device is None:
        device = labels.device 
    
    input_points = [dict() for _ in range(batchsize)]

    organ_id_range = list(range(1, n_categories + 1))
    if single_prompt:
        random.shuffle(organ_id_range)

    for i in range(batchsize):
        # for every image in the batch
        for organ_id in organ_id_range:
            # for every organ
            slice = labels[i] == organ_id 

            if torch.any(slice):
                input_points[i][organ_id] = sample_points(slice, n_points, device)

                if single_prompt:
                    break
    
    return input_points

def sample_points(slice, n_points, device):
    segmentation = torch.where(slice) # tuple[tensor, tensor]
    n_pixels = len(segmentation[0])
    selected = []

    while len(selected) < n_points:
        randint = torch.randint(0, n_pixels, (1,)).item()

        if n_pixels < n_points or randint not in selected:
            selected.append(randint)

    xs = segmentation[1][selected]
    ys = segmentation[0][selected]

    return torch.vstack([xs, ys]).T.to(device)

def box_prompt(labels : torch.Tensor, device=None, single_prompt=False):
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

    organ_id_range = list(range(1, n_categories + 1))
    if single_prompt:
        random.shuffle(organ_id_range)

    for i in range(batchsize):
        # for every image in the batch
        for organ_id in organ_id_range:
            # for every organ
            slice = labels[i] == organ_id

            if torch.any(slice):
                input_boxes[i][organ_id] = bounding_box(slice, device)

                if single_prompt:
                    break
    
    return input_boxes


def bounding_box(slice, device):
    segmentation = torch.where(slice)

    xmin = torch.min(segmentation[1])
    xmax = torch.max(segmentation[1])
    ymin = torch.min(segmentation[0])
    ymax = torch.max(segmentation[0])

    return torch.tensor([xmin, ymin, xmax, ymax], device=device)
