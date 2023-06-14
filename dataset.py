import nibabel as nib
import json
import os 
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils 

import warnings 
warnings.filterwarnings('ignore') 

class BTCV2DSliceDataset(Dataset):
    """ BTCV 2D Slice Dataset """

    def __init__(self, root_dir, json_file, 
                 type, preprocess=None, transform=None):
        """
        参数：
        root_dir:  数据所在的根目录;
        json_file: json文件路径。json文件中应当包含训练/测试数据集具体划分方式;
        type: 字符串。"training" 或 "validation";
        preprocess: 用于预处理读入的数据;
        transform: __getitem__中调用。用于(可能的)数据增强。
        """
        self.transform = transform

        with open(json_file, 'r') as f:
            self.dataset_info = json.load(f)[type]
        
        self.images = []
        self.labels = []

        for item in self.dataset_info:
            self.images.append(nib.load(
                                    os.path.join(root_dir, item['image'])
                                    ).get_fdata())
            self.labels.append(nib.load(
                                    os.path.join(root_dir, item['label'])
                                    ).get_fdata())
        
        self.images = np.concatenate(self.images, axis=2).transpose((2, 0, 1)) # shape: [N, H, W]
        self.labels = np.concatenate(self.labels, axis=2).transpose((2, 0, 1)) # shape: [N, H, W]

        if preprocess:
            self.images, self.labels = preprocess(self.images, self.labels)
        

    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = { 'image': self.images[idx],
                   'label': self.labels[idx] }

        if self.transform:
            sample = self.transform(sample)
        
        return sample

def gray2rgb(images : np.ndarray):
    # images shape: [N, H, W]
    # output shape: [N, H, W, 3] 
    return np.repeat(images[:, :, :, None], 3, axis=3)
    
def cvt2uint8(images : np.ndarray):
    # images shape: [N, H, W]
    im_max, im_min = np.max(images, axis=(1, 2), keepdims=True), \
                     np.min(images, axis=(1, 2), keepdims=True)

    images = np.array((images - im_min) / (im_max - im_min) * 256, 
                       dtype = np.uint8)

    return images

def remove_pure_background(images, labels):
    # images shape: [N, H, W]
    # labels shape: [N, H, W]
    # output shape: images: [M, H, W]
    #               labels: [M, H, W]
    label_sum = np.sum(labels, axis=(1,2))
    selection = label_sum > 0
    return images[selection], labels[selection]

def to_uint8_rgb(images, labels):
    # images shape: [N, H, W]
    # labels shape: [N, H, W]
    # output shape: images [N, H, W, 3]
    #               labels [N, H, W]
    return gray2rgb(cvt2uint8(images)), np.array(labels, dtype=np.uint8)