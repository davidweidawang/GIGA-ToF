'''
Author: Jin Zeng
Date: 2025-01-03 
LastEditTime: 2025-01-20
Description: GIGA for iToF denoising
'''

import os
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def print_info(arr, name=""):
    print(name, arr.shape, arr.min(), arr.max(), arr.mean())


def sqrt_ldr(correlations):
    tof_conf = np.abs(correlations[0,:,:]) + np.abs(correlations[1,:,:]) 
    tof_conf_l = 16 * np.sqrt(tof_conf + 36) - 96
    tof_conf[tof_conf==0] = 1
    i_tmp = tof_conf_l * correlations[0,:,:] / tof_conf
    q_tmp = tof_conf_l * correlations[1,:,:] / tof_conf
    
    return np.stack((i_tmp, q_tmp), axis=0)


def load_depth(scene):
    """
    :param scene: path of depth
    :return: depth in unit meter
    """
    depth = np.load(scene)
    data_expanded = np.expand_dims(depth.astype(np.float32), 0)
    return data_expanded


def load_raw(scene, sqrt_in=False):
    raw = np.load(scene)

    tof_IQ_40 = np.stack((raw[0], raw[1]), axis=0)
    tof_IQ_30 = np.stack((raw[3], raw[4]), axis=0)
    tof_IQ_58 = np.stack((raw[6], raw[7]), axis=0)

    if sqrt_in:
        tof_IQ_40 = sqrt_ldr(tof_IQ_40)
        tof_IQ_30 = sqrt_ldr(tof_IQ_30)
        tof_IQ_58 = sqrt_ldr(tof_IQ_58)
    
    tof_IQs = np.concatenate((tof_IQ_30, tof_IQ_40, tof_IQ_58), axis=0)
    tof_IQs = np.nan_to_num(tof_IQs, nan=0, posinf=0, neginf=0)
    
    scaler = 500
    tof_IQs = tof_IQs / scaler      
    return tof_IQs.astype(np.float32)


class pbrt_dv_Dataset(Dataset):
    """ 
        Dataset loader
        load all data at once for better time efficiency 
    """
    def __init__(self, root="./dataset", mode='train', sqrt_ldr=True, transform=None):
        """
        Args:
            img_dir (string): Path to the image files with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.sqrt_ldr = sqrt_ldr
        self.mode = mode

        if mode != 'train' and mode != 'test':
            raise NotImplementedError  

        self.list = []
        self.gts = []
        self.noises = []
        self.noises_prev = []
        self.gt_depths = []
        
        self.transform = transform    

        if mode == 'train':
            file_list = os.path.join(root, "list/train.txt")
        elif mode == 'test':
            file_list = os.path.join(root, "list/test.txt")
        
        with open(file_list, 'r') as f:
            for line in f:
                self.list.append(line.strip('\n'))
        
        pbar = tqdm(self.list, desc=f"Loading {mode} data")
        for idx in pbar:
            for frame_id in range(2, 251):  # start from the 2nd frame
                
                pbar.set_postfix(frame=f"{idx}/{frame_id}")
                prev_frame_id = frame_id - 1  # previous frame

                noise = load_raw(os.path.join(root, "noise", idx, f"{frame_id}.npy"))
                noise_prev = load_raw(os.path.join(root, "noise", idx, f"{prev_frame_id}.npy"))
                gt = load_raw(os.path.join(root, "gt", idx, f"{frame_id}.npy"))
                gt_depth = load_depth(os.path.join(root, "gt_depth", idx, f"{frame_id}.npy"))

                noise_tensor = torch.from_numpy(noise).float()
                noise_prev_tensor = torch.from_numpy(noise_prev).float()
                gt_tensor = torch.from_numpy(gt).float()
                depth_tensor = torch.from_numpy(gt_depth).float()

                self.noises.append(noise_tensor)
                self.noises_prev.append(noise_prev_tensor)
                self.gts.append(gt_tensor)
                self.gt_depths.append(depth_tensor)

    def __len__(self):
        return len(self.list) * 249
    
    def __getitem__(self, index):
        return self.noises[index], self.noises_prev[index], self.gts[index], self.gt_depths[index]


if __name__ == '__main__':
    batch_size = 10
    root = "./dataset"
    train_data = pbrt_dv_Dataset(root, mode="train")
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, drop_last=True)
    print(train_dataloader.__len__())

    for i, mini_batch in enumerate(train_dataloader):
        noise, gt, gt_depth = mini_batch
        if torch.all(gt_depth.eq(0)):
            print(i, "gt depth all 0")
        if torch.any(torch.isnan(noise)):
            print(i, "noise nan")
        if torch.any(torch.isnan(gt)):
            print(i, "gt nan")
        if torch.any(torch.isnan(gt_depth)):
            print(i, "gt depth nan")
