import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything.utils.transforms import ResizeLongestSide

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def prepare_image(image, transform, device):
    #print(image.shape)
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device) 
    return image.permute(2, 0, 1).contiguous()


def batched_input_gen(images, prompts, types, sam):
    """
    参数：
    images:  torch.Tensor [B, H, W, 3] 
    prompts: list of dictionaries of length B
             for box prompts: organ_id -> torch.Tensor[xmin, ymin, xmax, ymax]
             for point prompts: organ_id -> torch.Tensor [N,2]
    types:   list of strings indicating prompt types 
             each string can be 'box' or 'point'
    返回：
    batched_input: list of dictionaries with keys 
             'image', 'original size', 'point_coords', 'point_labels', 'boxes'
    """
    batch_size = images.shape[0]
    batched_input = []
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    for img_idx in range(batch_size):
        input = {'image' : prepare_image(images[img_idx].numpy(), resize_transform, sam.device), 
                 'original_size' : images.shape[1:3]}

        if types[img_idx] == 'box':
            bboxes = prompts[img_idx] # dict of organ_id -> bbox 
            input['boxes'] = resize_transform.apply_boxes_torch(torch.vstack(list(bboxes.values())), 
                                                                                 input['original_size'])
        elif types[img_idx] == 'point':
            points = prompts[img_idx] # dict of organ_id -> torch.Tensor[N, 2]
            points = torch.stack(list(points.values())) # torch.Tensor [B, N, 2]
            
            input['point_coords'] = resize_transform.apply_coords_torch(points, input['original_size'])
            input['point_labels'] = torch.ones(points.shape[:2], dtype=points.dtype, 
                                               device=points.device)
        else:
            raise NotImplementedError
        
        batched_input.append(input)
    
    return batched_input