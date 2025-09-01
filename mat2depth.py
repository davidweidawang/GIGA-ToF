import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

from utils import load_depth_from_mat


if __name__ == '__main__':  
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default="results")
    parser.add_argument('--version', type=str)
    parser.add_argument('--list_path', type=str)
    parser.add_argument('--visualize', type=bool, default=True)

    args = parser.parse_args()

    root = args.root
    version = args.version
    list_path = args.list_path
    visualize = args.visualize

    predict_list = []
    with open(list_path, 'r') as f:
        for line in f:
            path = line.strip('\n')
            predict_list.append(path)
    
    pbar = tqdm(predict_list, desc=f"Mat2Depth")
    for scene in pbar:
        mat_dir = f"{root}/{version}/depth_mats/{scene}"
        depth_dir = f'{root}/{version}/depth/{scene}'
        assert(os.path.exists(mat_dir))
        os.makedirs(depth_dir, exist_ok=True)

        if visualize:
            png_dir = f'{root}/{version}/png/{scene}'
            os.makedirs(png_dir, exist_ok=True)
        
        for frame_id in range(2, 251):
            pbar.set_postfix(frame=f"{scene}/{frame_id}")
            depth = load_depth_from_mat(f"{mat_dir}/{frame_id}.mat")
            np.save(f"{depth_dir}/{frame_id}.npy", depth)

            if visualize:
                cv2.imwrite(f"{png_dir}/{frame_id}.png", depth * 25)
