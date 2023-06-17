import torch 
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tqdm
from utils import batched_input_gen
from segment_anything.modeling.sam import Sam
from sam_btcv import SamBTCV
from prompt_gen import *
from metric import *

@torch.no_grad()
def test_model(validation_loader : DataLoader,
               prompt_generate,  # function (labels, device) -> prompt (list of dictionaries)
               prompt_type : str,
               sam : Sam):
    total_intersection = dict()  # organ_id -> total intersect pixels
    total_mask_pixels = dict()   # organ_id -> total mask pixels

    # Dice = 2 * intersect / total mask pixels 
    # mDice = Average Dice over all categories

    for batched_data in validation_loader:
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

@torch.no_grad()
def validation_loss(validation_set : Dataset, 
                    model : SamBTCV, 
                    p_prompts = [1/3, 1/6, 1/6, 1/3]):
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    device = model.device

    random_p = np.random.uniform(0, 1, len(validation_loader)).reshape((-1, 1))
    cum_p = np.cumsum(p_prompts)
    prompt_indices = ((random_p - cum_p) > 0).sum(axis=1)
    prompt_generators = [lambda labels, device: box_prompt(labels, device, True), 
                     lambda labels, device: point_prompt(labels, 2, device, True), 
                     lambda labels, device: point_prompt(labels, 3, device, True), 
                     lambda labels, device: point_prompt(labels, 1, device, True)]
    prompt_types = ['box'] + ['point'] * 3

    focal_losses = []
    dice_losses = []
    mse_losses = []
    cross_entropy_losses = []

    for i, data in enumerate(validation_loader):
        imgs, labels = data['image'], data['label']
        prompt_idx = prompt_indices[i]
        prompt = prompt_generators[prompt_idx](labels, model.device)
        gt_label = list(prompt[0].keys())[0]
        batched_input = batched_input_gen(imgs, prompt, [prompt_types[prompt_idx]] * 1, model)
        batched_output = model(batched_input, False)[0]

        pred_mask = batched_output['masks']
        iou_prediction = batched_output['iou_predictions']
        pred_logits = model.postprocess_masks(batched_output['low_res_logits'], batched_input[0]['image'].shape[-2:], 
                                              batched_input[0]['original_size'])
        if model.requires_classification:
            pred_class = batched_output['class_predictions']
            cross_entropy_losses.append(F.cross_entropy(pred_class[0][0], torch.tensor(gt_label, device=device)))

        gt_mask = (labels == gt_label).to(device)
        mse_losses.append((IoU(pred_mask[0], gt_mask) - iou_prediction) ** 2)
        gt_mask = torch.where(gt_mask, 1.0 , 0.0)
        focal_losses.append(FocalLoss(pred_logits, gt_mask))
        dice_losses.append(DiceLoss(pred_logits, gt_mask))

    mse_loss = torch.mean(torch.stack(mse_losses))
    focal_loss = torch.mean(torch.stack(focal_losses))
    dice_loss = torch.mean(torch.stack(dice_losses))
    cross_entropy_loss = torch.mean(torch.stack(cross_entropy_losses)) if model.requires_classification \
                                                                               else torch.tensor([0], device=device)

    total_loss = mse_loss + 20 * focal_loss + dice_loss + cross_entropy_loss
    
    return total_loss.item()
    