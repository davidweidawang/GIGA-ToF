'''
Author: Jin Zeng, Changyong He
LastEditTime: 2025-01-20
Description: evaluate depth error, output RMSE, MAE, iRMSE, iMAE, delta
'''
import os
import argparse
from tqdm import tqdm

import numpy as np
import cv2


def warp_depth(depth, flow):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    u_new = u + flow[0]
    v_new = v + flow[1]
    
    u0 = np.floor(u_new).astype(np.int32)
    v0 = np.floor(v_new).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1
    
    u0 = np.clip(u0, 0, W - 1)
    u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)
    
    wa = (u1 - u_new) * (v1 - v_new)
    wb = (u1 - u_new) * (v_new - v0)
    wc = (u_new - u0) * (v1 - v_new)
    wd = (u_new - u0) * (v_new - v0)
    
    Ia = depth[v0, u0]
    Ib = depth[v1, u0]
    Ic = depth[v0, u1]
    Id = depth[v1, u1]
    
    warped = wa * Ia + wb * Ib + wc * Ic + wd * Id
    return warped


def compute_tepe_from_depths(gt_prev, gt_curr, pred_prev, pred_curr, flow):
    """
    Compute TEPE (Temporal Edge Preserving Error) from depth maps.

    Args:
        gt_prev:  Ground-truth depth at time t-1
        gt_curr:  Ground-truth depth at time t
        pred_prev: Predicted depth at time t-1
        pred_curr: Predicted depth at time t
        flow:     Optical flow from t-1 to t (used to warp depths)

    Returns:
        tepe: Scalar TEPE value.
    """
    mask = (gt_curr > 0.001) & (gt_curr < 10) & (pred_curr > 0)
    
    # Warp the depth map at t-1 to the t frame using optical flow
    warped_gt_prev = warp_depth(gt_prev, flow)
    warped_pred_prev = warp_depth(pred_prev, flow)
    
    # Compute inter-frame depth differences (warped prev vs current)
    diff_gt = warped_gt_prev - gt_curr
    diff_pred = warped_pred_prev - pred_curr
    
    # TEPE: mean absolute difference between the two differences
    tepe = np.mean(np.abs(diff_gt[mask] - diff_pred[mask]))
    return tepe

def loss(pred, gt):
    t_valid = 0.01
    t_max = 10 # max range for 30MHz

    pred[pred >= t_max] = 0
    pred[pred <= 0] = 0

    pred_inv = 1.0 / (pred + 1e-8)
    gt_inv = 1.0 / (gt + 1e-8)

    mask = (gt > t_valid) & (gt < t_max) & (pred > 0)
    num_valid = mask.sum()
    if num_valid == 0:
        num_valid = 1

    pred = pred[mask]
    gt = gt[mask]

    pred_inv = pred_inv[mask]
    gt_inv = gt_inv[mask]

    pred_inv[pred <= t_valid] = 0.0
    gt_inv[gt <= t_valid] = 0.0

    # RMSE / MAE
    diff = pred - gt
    diff_abs = np.abs(diff)
    diff_sqr = np.power(diff, 2)

    rmse = diff_sqr.sum() / num_valid
    rmse = np.sqrt(rmse)

    mae = diff_abs.sum() / num_valid

    # iRMSE / iMAE
    diff_inv = pred_inv - gt_inv
    diff_inv_abs = np.abs(diff_inv)
    diff_inv_sqr = np.power(diff_inv, 2)

    irmse = diff_inv_sqr.sum() / num_valid
    irmse = np.sqrt(irmse)

    imae = diff_inv_abs.sum() / num_valid

    # Rel
    rel = diff_abs / (gt + 1e-8)
    rel = rel.sum() / num_valid

    # delta
    r1 = gt / (pred + 1e-8)
    r2 = pred / (gt + 1e-8)
    ratio = np.maximum(r1, r2)

    del_1 = (ratio < 1.25).astype('float32')
    del_2 = (ratio < 1.25 ** 2).astype('float32')
    del_3 = (ratio < 1.25 ** 3).astype('float32')

    del_1 = del_1.sum() / num_valid
    del_2 = del_2.sum() / num_valid
    del_3 = del_3.sum() / num_valid

    result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3]

    return result


def main(args):
    # args
    in_path = args.in_path      # result depth
    gt_path = args.gt_path      # ideal depth
    flow_path = args.flow_path  # optic_flow
    version = args.version
    out_path = args.save_dir    # save metric
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    loss_rmse = []
    loss_mae = []
    loss_irmse = []
    loss_imae = []
    loss_rel = []
    loss_del_1 = []
    loss_del_2 = []
    loss_del_3 = []
    tepes = []

    # nimg_list = sorted(os.listdir(in_path))
    scenes = []
    list_path = args.list_path
    with open(list_path, 'r') as f:
        for line in f:
            path = line.strip('\n')
            scenes.append(path)

    pbar = tqdm(scenes, desc=f"Evaluating")
    for _, scene in enumerate(pbar, 0):
        for frame_id in range(3, 251): # time step 8
            pbar.set_postfix(frame=f"{scene}/{frame_id}")
            if scene == "white-room/7" and frame_id == 2:   # white-room第1帧和第2帧差距太大
                continue

            gt_prev = np.load(os.path.join(gt_path, scene, f"{frame_id - 1}.npy"))
            pred_prev = np.load(os.path.join(in_path, scene, f"{frame_id - 1}.npy"))

            pred = np.load(os.path.join(in_path, scene, f"{frame_id}.npy"))
            gt = np.load(os.path.join(gt_path, scene, f"{frame_id}.npy"))
            flow = np.load(os.path.join(flow_path, scene, f"{frame_id}.npy"))
                  
            # tepe
            tepe = compute_tepe_from_depths(gt_prev, gt, pred_prev, pred, flow)
            tepes.append(tepe)

            # total loss
            loss_list = loss(pred, gt)
            loss_rmse.append(loss_list[0])
            loss_mae.append(loss_list[1])
            loss_irmse.append(loss_list[2])
            loss_imae.append(loss_list[3])
            loss_rel.append(loss_list[4])
            loss_del_1.append(loss_list[5])
            loss_del_2.append(loss_list[6])
            loss_del_3.append(loss_list[7])

    # save
    rmse_mean = sum(loss_rmse) / len(loss_rmse)
    mae_mean = sum(loss_mae) / len(loss_mae)
    irmse_mean = sum(loss_irmse) / len(loss_irmse)
    imae_mean = sum(loss_imae) / len(loss_imae)
    rel_mean = sum(loss_rel) / len(loss_rel)
    del_1_mean = sum(loss_del_1) / len(loss_del_1)
    del_2_mean = sum(loss_del_2) / len(loss_del_2)
    del_3_mean = sum(loss_del_3) / len(loss_del_3)
    tepe_mean = sum(tepes) / len(tepes)

    print("rmse_mean, mae_mean, irmse_mean, imae_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(rmse_mean, mae_mean,
                                                                                               irmse_mean, imae_mean))
    print("rel_mean, del_1_mean, del_2_mean, del_3_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f}".format(rel_mean, del_1_mean,
                                                                                                 del_2_mean,
                                                                                                 del_3_mean))
    print("TEPE: {0:.4f}".format(tepe_mean))
    with open(f"{out_path}/result_metrics_{version}.txt", "w") as text_file:
        text_file.write(
            "rmse_mean, mae_mean, irmse_mean, imae_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f} \n".format(rmse_mean, mae_mean,
                                                                                                    irmse_mean,
                                                                                                    imae_mean))
        text_file.write(
            "rel_mean, del_1_mean, del_2_mean, del_3_mean: {0:.4f} {1:.4f} {2:.4f} {3:.4f} \n".format(rel_mean, del_1_mean,
                                                                                                   del_2_mean,
                                                                                                   del_3_mean))
        text_file.write("TEPE: {0:.4f}".format(tepe_mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-in", "--in_path", type=str, default='./predict_result2',
        help="pred depth directory")

    parser.add_argument(
        "-gt", "--gt_path", type=str, default='../DVToF/ideal_depth',
        help="GT depth directory")
    
    parser.add_argument(
        "-flow", "--flow_path", type=str, default='../DVToF/optic_flow',
        help="optic flow directory")

    parser.add_argument(
        "-out", "--save_dir", type=str, default='./result_metrics',
        help="result metrics directory")
    
    parser.add_argument(
        "-v", "--version", type=str,
        help="ideal depth directory")

    parser.add_argument('--list_path', type=str)

    args = parser.parse_args()
    main(args)

