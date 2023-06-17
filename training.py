import os
import torch
import torch.nn.functional as F
import numpy as np
from metric import *
from prompt_gen import *
from utils import *
from testing import *
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
    cross_entropy_losses = []

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
        if model.requires_classification:
            pred_class = batched_output['class_predictions']
            cross_entropy_losses.append(F.cross_entropy(pred_class[0][0], torch.tensor(gt_label, device=device)))

        gt_mask = (labels == gt_label).to(device)
        mse_losses.append((IoU(pred_mask[0], gt_mask) - iou_prediction) ** 2)
        gt_mask = torch.where(gt_mask, 1.0 , 0.0)
        focal_losses.append(FocalLoss(pred_logits, gt_mask))
        dice_losses.append(DiceLoss(pred_logits, gt_mask))

        if (i + 1) % batch_size == 0:
            mse_loss = torch.mean(torch.stack(mse_losses))
            focal_loss = torch.mean(torch.stack(focal_losses))
            dice_loss = torch.mean(torch.stack(dice_losses))
            cross_entropy_loss = torch.mean(torch.stack(cross_entropy_losses)) if model.requires_classification \
                                                                               else torch.tensor([0], device=device)

            total_loss = mse_loss + 20 * focal_loss + dice_loss + cross_entropy_loss
            print('mse loss:', mse_loss.item(), flush=True)
            print('focal loss:', focal_loss.item(), flush=True)
            print('dice loss:', dice_loss.item(), flush=True)
            print('cross entropy loss:', cross_entropy_loss.item(), flush=True)
            print('total loss:', total_loss.item(), flush=True)
            total_loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            dice_losses, focal_losses, mse_losses, cross_entropy_losses = [], [], [], []
    
    if len(mse_losses) != 0:
        mse_loss = torch.mean(torch.stack(mse_losses))
        focal_loss = torch.mean(torch.stack(focal_losses))
        dice_loss = torch.mean(torch.stack(dice_losses))
        cross_entropy_loss = torch.mean(torch.stack(cross_entropy_losses)) if model.requires_classification else 0

        total_loss = mse_loss + 20 * focal_loss + dice_loss + cross_entropy_loss
        print('total_loss:', total_loss.item())
        total_loss.backward()
        optimizer.step()

        optimizer.zero_grad()


def train_model(train_set : Dataset, 
                validation_set : Dataset, 
                optimizer : Optimizer, 
                lr_scheduler : LRScheduler, 
                model : SamBTCV,
                best_model_root : str = './', 
                batch_size : int = 32, 
                n_epochs : int = 50
                ):
    best_mDice = 0

    for epoch in range(n_epochs):
        print('Training Epoch', epoch, flush=True)

        model.train()
        train_one_epoch(train_set, optimizer, model, batch_size)
        lr_scheduler.step()
        model.eval()

        validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
        _, mDice_p = test_model(validation_loader, lambda labels, device: point_prompt(labels, 1, device), 'point', model)
        _, mDice_b = test_model(validation_loader, lambda labels, device: box_prompt(labels, device), 'box', model)

        mean_dice = (mDice_p + mDice_b) / 2

        if mean_dice > best_mDice:
            best_mDice = mean_dice
            torch.save(model.state_dict(), os.path.join(best_model_root, 'best_model.pth'))
        