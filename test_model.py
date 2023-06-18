from sam_btcv import SamBTCV 
from segment_anything import sam_model_registry
from prompt_gen import *
from testing import test_model
from dataset import BTCV2DSliceDataset, to_uint8_rgb, remove_pure_background, to_tensor
import torch
from torch.utils.data import DataLoader

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    sam = sam_model_registry['default']('./sam_vit_h_4b8939.pth')
    model = SamBTCV(sam)
    model.load_state_dict(torch.load('./no_cls_b32_ada/best_model.pth'))
    model.to(device)

    preprocess = lambda images, labels: to_tensor(*to_uint8_rgb(*remove_pure_background(images, labels)))

    validation_set = BTCV2DSliceDataset(root_dir='./data', 
                                        json_file='./data/dataset_0.json', 
                                        type='validation', 
                                        preprocess=preprocess)
    
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: box_prompt(labels, device), 'box', model) 
    print(dice)
    print(mDice) 
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: point_prompt(labels, 1, device), 'point', model)
    print(dice)
    print(mDice)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: point_prompt(labels, 2, device), 'point', model)
    print(dice)
    print(mDice)
    validation_loader = DataLoader(validation_set, batch_size=1, shuffle=False)
    dice, mDice = test_model(validation_loader, lambda labels, device: point_prompt(labels, 3, device), 'point', model)
    print(dice)
    print(mDice)


if __name__ == '__main__':
    main()