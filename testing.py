import torch 
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from utils import batched_input_gen
from segment_anything.modeling.sam import Sam

def test_model(validation_loader : DataLoader,
               prompt_generate,  # function (labels, device) -> prompt (list of dictionaries)
               prompt_type : str,
               sam : Sam):
    total_intersection = dict()  # organ_id -> total intersect pixels
    total_mask_pixels = dict()   # organ_id -> total mask pixels

    # Dice = 2 * intersect / total mask pixels 
    # mDice = Average Dice over all categories

    for batched_data in tqdm.tqdm(validation_loader):
        images, labels = batched_data['image'], batched_data['label'] # labels transferred to mps will cause strange errors
        batch_size = images.shape[0]
        prompts = prompt_generate(labels, device=sam.device)
        batched_input = batched_input_gen(images, prompts, [prompt_type] * batch_size, sam)
        batched_output = sam(batched_input, multimask_output=False)

        for img_idx in range(batch_size):
            # for every input image
            output = batched_output[img_idx]
            for box_idx, gt_label in enumerate(prompts[img_idx].keys()):
                # for every bounding box in the image
                gt_mask = (labels[img_idx] == gt_label).cpu()
                pred_mask = output['masks'][box_idx].cpu()

                intersection = torch.sum(torch.logical_and(gt_mask ,pred_mask)).item()
                total_pixels = torch.sum(gt_mask).item() + torch.sum(pred_mask).item()

                total_intersection.setdefault(gt_label, 0)
                total_mask_pixels.setdefault(gt_label, 0)

                total_intersection[gt_label] += intersection
                total_mask_pixels[gt_label] += total_pixels 

    dice = dict([(organ_id , 2 * total_intersection[organ_id] / total_mask_pixels[organ_id]) for organ_id in total_intersection])
    mDice = np.mean(list(dice.values()))

    return dice, mDice