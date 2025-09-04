##################################
# Train GIGA-ToF with DVToF dataset
# Input: kinect iq
# Output: depth
##################################

import os
import time
import argparse

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from giga.GIGA import GIGAToF
from giga.DVToF_dataloader import load_raw

import warnings
warnings.filterwarnings("ignore")


def sqrt_hdr(correlations):
    correlations = correlations[0].detach().numpy()
    correlations = correlations * 500

    tof_conf = np.abs(correlations[0, :, :]) + np.abs(correlations[1, :, :])
    tof_conf_h = (tof_conf / 16 + 6) ** 2 - 36

    tof_conf[tof_conf == 0] = 1
    iq_0 = tof_conf_h * correlations[0, :, :] / tof_conf
    iq_1 = tof_conf_h * correlations[1, :, :] / tof_conf

    return np.stack((iq_0, iq_1), axis=0)


def get_input(scene):
    tof_raw_IQ = load_raw(scene, sqrt_in=True)
    tof_raw_IQ_tensor = torch.from_numpy(tof_raw_IQ).float()  # [6, 424, 512]

    return tof_raw_IQ_tensor

def predict_single_scene(scene, scene_pre, device, model):
    with torch.no_grad():
        raw_IQ = get_input(scene)
        raw_IQ_pre = get_input(scene_pre)

        raw_IQ = raw_IQ.to(device)
        raw_IQ = raw_IQ.unsqueeze(0)
        raw_IQ_pre = raw_IQ_pre.to(device)
        raw_IQ_pre = raw_IQ_pre.unsqueeze(0)

        t0 = time.time()
        out_0, mu0,_,_ = model(raw_IQ[:, 0:2, :, :], raw_IQ_pre[:, 0:2, :, :])
        out_1, mu1,_,_ = model(raw_IQ[:, 2:4, :, :], raw_IQ_pre[:, 2:4, :, :])
        out_2, mu2,_,_ = model(raw_IQ[:, 4:6, :, :], raw_IQ_pre[:, 4:6, :, :])
        
        t1 = time.time()

    # sqrt in
    out_0 = sqrt_hdr(out_0.cpu())
    out_1 = sqrt_hdr(out_1.cpu())
    out_2 = sqrt_hdr(out_2.cpu())

    outputs = np.concatenate((out_0, out_1, out_2), axis=0)

    outputs_mu = np.concatenate((mu0.cpu().detach().numpy(), mu1.cpu().detach().numpy(), mu2.cpu().detach().numpy()),
                                axis=0)

    return outputs, outputs_mu, t1-t0


def predict(args):
    cudaid = "cuda:" + str(args.dev)
    device = torch.device(cudaid)

    # args
    raw_dir = args.train_path
    out_dir = args.destination
    out_mu_dir = args.destination_mu
    list_path = args.list_path
    model_path = args.model
    if not os.path.exists(raw_dir):
        print(f"Dataset path '{raw_dir}' does not exist!")
        raise FileNotFoundError
    if not os.path.exists(model_path):
        print(f"Model '{model_path}' does not exist!")
        raise FileNotFoundError
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(out_mu_dir):
        os.makedirs(out_mu_dir)
    print(device, model_path)

    # load file list
    predict_list = []
    with open(list_path, 'r') as f:
        for line in f:
            path = line.strip('\n')
            predict_list.append(path)

    # model
    gspn = GIGAToF()
    checkpoint = torch.load(model_path)
    gspn.load_state_dict(checkpoint['model_state_dict'])
    gspn.to(device)
    gspn = gspn.eval()

    t_total = 0
    num_sample = len(predict_list) * 249#250
    
    pbar = tqdm(predict_list, desc=f"Predicting")
    for scene in pbar:
        os.makedirs(f"{out_dir}/{scene}", exist_ok=True)
        os.makedirs(f"{out_mu_dir}/{scene}", exist_ok=True)

        for frame_id in range(2, 251):
            pbar.set_postfix(frame=f"{scene}/{frame_id}")
            # noisy data
            # out_iq, out_mu, t_last = predict_single_scene(f"{raw_dir}/barron/{scene}/{frame_id}.npy", f"{raw_dir}/barron/{scene}/{frame_id-1}.npy", device, gspn)
            out_iq, out_mu, t_last = predict_single_scene(f"{raw_dir}/noise/{scene}/{frame_id}.npy", f"{raw_dir}/noise/{scene}/{frame_id-1}.npy", device, gspn)\

            np.save(f"{out_dir}/{scene}/{frame_id}.npy", out_iq)
    
    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))
    print("[ End Predicting ]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--dev', type=int, default=0, help='device id')

    parser.add_argument("-in", "--train_path", type=str, default='./dataset/noise_IQ', help="Train set directory")
    parser.add_argument("-ls", "--list_path", type=str, default='./dataset/list/test.txt', help='Path to the test list file')
    parser.add_argument("-out", "--destination", type=str, default='./predict_result_iq2', help="Output destination.")
    parser.add_argument("-out_mu", "--destination_mu", type=str, default='./predict_result_mu2',
                        help="Output destination.")
    parser.add_argument("-m", "--model", type=str, default='models/checkpoint_best.pth',
                        help="Path to the trained GIGAToF.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    predict(args)
