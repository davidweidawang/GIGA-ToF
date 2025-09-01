import os
import numpy as np
from scipy import ndimage
from scipy.io import savemat, loadmat


def print_info(arr, name=""):
    print(name, arr.shape, arr.min(), arr.max(), arr.mean())


def raw2corr(raw, save_path):
    """
    convert raw generated from pbrt to correlation for further MATLAB process 
    @param: raw: 
        dtype: np.ndarray, 
        shape: (9, h, w),  Q40 I40 A40 Q30 I30 A30 Q58 I58 A58
    @output: corr_imgs:
        dtype: np.ndarray,
        shape: (6, h, w), Q40 Q30 Q58 I40 I30 I58
    """
    corr_imgs = np.stack([
        raw[0], raw[3], raw[6],
        raw[1], raw[4], raw[7]
    ], axis=0)

    savemat(save_path, { "corr_imgs": corr_imgs })


def IQ2corr(IQ, save_path):
    """
    convert IQ predicted by GIGAToF to correlation for further MATLAB process 
    @param: IQ: 
        dtype: np.ndarray, 
        shape: (6, h, w),  I30 Q30 I40 Q40 I58 Q58
    @output: corr_imgs:
        dtype: np.ndarray,
        shape: (6, h, w), Q40 Q30 Q58 I40 I30 I58
    """
    corr_imgs = np.stack([
        IQ[3], IQ[1], IQ[5],
        IQ[2], IQ[0], IQ[4]
    ], axis=0)

    savemat(save_path, { "corr_imgs": corr_imgs })


def distance_transform(depth, mask):
    depth = np.array(depth, dtype=float)
    mask = np.array(mask, dtype=bool)
    
    if depth.shape != mask.shape:
        raise ValueError
    
    out_depth = depth.copy()
    
    known_values = depth[~mask]
    
    if len(known_values) == 0:
        return depth
    
    _, indices = ndimage.distance_transform_edt(
        mask,
        return_indices=True  
    )

    out_depth[mask] = depth[indices[0][mask], indices[1][mask]]
    
    return out_depth.astype(np.float32)


def load_depth_from_mat(depth_path):
    """
    load the depth file processed via MATLAB
    """
    depth = loadmat(depth_path)["depths"]
    depth = np.nan_to_num(depth, 0)
    mask = ((depth < 0.1) | (depth > 10)).astype(np.uint8)

    return distance_transform(depth, mask)
    # return depth
