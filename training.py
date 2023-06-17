import torch
import torch.nn as nn
import numpy as np
from metric import *
from prompt_gen import *
from utils import *
from dataset import BTCV2DSliceDataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from sam_btcv import SamBTCV 


def train_one_epoch(train_set : Dataset, 
                    optimizer : Optimizer,
                    model : SamBTCV, 
                    batch_size : int = 32,
                    p_prompts = [1/3, 1/6, 1/6, 1/3]):
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    device = model.device

    random_p = np.random.uniform(0, 1, len(train_loader)).reshape((-1, 1))
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

    optimizer.zero_grad()

    for i, data in enumerate(train_loader):
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

        gt_mask = (labels == gt_label).to(device)
        mse_losses.append((IoU(pred_mask[0], gt_mask) - iou_prediction) ** 2)
        gt_mask = torch.where(gt_mask, 1.0 , 0.0)
        focal_losses.append(FocalLoss(pred_logits, gt_mask))
        dice_losses.append(DiceLoss(pred_logits, gt_mask))

        if (i + 1) % batch_size == 0:
            mse_loss = torch.mean(torch.stack(mse_losses))
            focal_loss = torch.mean(torch.stack(focal_losses))
            dice_loss = torch.mean(torch.stack(dice_losses))

            total_loss = mse_loss + 20 * focal_loss + dice_loss
            print('total_loss:', total_loss)
            total_loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            dice_losses, focal_losses, mse_losses = [], [], []
    
    if len(mse_losses) != 0:
        mse_loss = torch.mean(torch.stack(mse_losses))
        focal_loss = torch.mean(torch.stack(focal_losses))
        dice_loss = torch.mean(torch.stack(dice_losses))

        total_loss = mse_loss + 20 * focal_loss + dice_loss
        print('total_loss:', total_loss.item())
        total_loss.backward()
        optimizer.step()

        optimizer.zero_grad()


def train_model():
    pass