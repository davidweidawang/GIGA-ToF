import os
import numpy as np
import argparse
from tqdm import tqdm

from utils import IQ2corr


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default="results")
    parser.add_argument('--version', type=str)
    parser.add_argument('--list_path', type=str)

    args = parser.parse_args()

    root = args.root
    version = args.version
    list_path = args.list_path

    predict_list = []
    with open(list_path, 'r') as f:
        for line in f:
            path = line.strip('\n')
            predict_list.append(path)
    
    pbar = tqdm(predict_list, desc=f"IQ2Corr")
    for _, scene in enumerate(pbar, 0):
        iq_dir = f"{root}/{version}/iq/{scene}"
        corr_dir = f'{root}/{version}/corr/{scene}'
        assert(os.path.exists(iq_dir))
        os.makedirs(corr_dir, exist_ok=True)

        pbar.set_postfix(scene=scene)

        for frame_id in range(2, 251):
            raw = np.load(f"{iq_dir}/{frame_id}.npy")
            IQ2corr(raw, f"{corr_dir}/{frame_id}.mat")
        
