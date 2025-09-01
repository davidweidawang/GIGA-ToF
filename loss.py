'''
Author: Jin Zeng, Yaxuan Chen
Date: 2023-04-13
LastEditTime: 2025-01-20
Description: loss function
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def iq2d(iq):
    theta = torch.atan2(iq[:,0:1,:,:], iq[:,1:2,:,:])
    theta[torch.lt(theta, 0)] = theta[torch.lt(theta, 0)]+2*np.pi
    depth = (theta / (2*np.pi) )
   
    return depth
    

class GLoss(nn.Module):
    def __init__(self, device, weight=0.05, alpha=0.3, lambda_attconf_reg=0.005, lambda_inter_sparse=0.005):
        super(GLoss, self).__init__()
        self.weight = weight
        self.max_range_iq = 0.15
        self.max_range_d = 1
        self.device = device

        # GIGA 模块损失超参数
        self.alpha = alpha  # IQ加权主损失比例
        self.lambda_attconf_reg = lambda_attconf_reg
        self.lambda_inter_sparse = lambda_inter_sparse

    def forward(self,
                out_0, out_1, out_2,
                ideal_IQ, ideal_d,
                inter_graph_0, attconf_0,
                inter_graph_1, attconf_1,
                inter_graph_2, attconf_2):

        # === GT拆分 ===
        ideal_IQ_0 = ideal_IQ[:, 0:2, :, :]
        ideal_IQ_1 = ideal_IQ[:, 2:4, :, :]
        ideal_IQ_2 = ideal_IQ[:, 4:6, :, :]

        # === mask 和裁剪用量 ===
        d_mask = (ideal_d != 0) & (ideal_d < 10)
        iq_mask = torch.cat([d_mask, d_mask], dim=1)
        other_d = torch.ones_like(ideal_d).to(self.device)
        other_iq = torch.ones_like(ideal_IQ_0).to(self.device)

        # # === 加权 IQ loss 工具函数 ===
        def weighted_iq_loss(pred, gt, att):
            abs_error = torch.abs(pred - gt)
            error_clip = torch.min(abs_error, self.max_range_iq * other_iq)
            att = att.expand_as(pred).detach()  # [B, 1, H, W] → [B, 2, H, W]
            weighted = att * error_clip
            unweighted = error_clip
            return (1 - self.alpha) * unweighted[iq_mask].mean() + self.alpha * weighted[iq_mask].mean()

        # === 正则项 ===
        def attconf_reg_loss(att):
            return (att.mean() - 0.5) ** 2

        def inter_sparse_loss(inter):
            entropy = -inter * torch.log(inter + 1e-6)
            return -entropy.sum(dim=1).mean()

        # === IQ loss 每阶段 ===
        iq_loss_0 = weighted_iq_loss(out_0, ideal_IQ_0, attconf_0)
        iq_loss_1 = weighted_iq_loss(out_1, ideal_IQ_1, attconf_1)
        iq_loss_2 = weighted_iq_loss(out_2, ideal_IQ_2, attconf_2)
        loss_sup = iq_loss_0 + iq_loss_1 + iq_loss_2

        # === Depth loss ===
        def d_loss_fn(pred_iq, gt_iq):
            d_pred = iq2d(pred_iq)
            d_gt = iq2d(gt_iq)
            return torch.min(torch.abs(d_pred - d_gt), self.max_range_d * other_d)[d_mask].mean()

        d_loss_0 = d_loss_fn(out_0, ideal_IQ_0)
        d_loss_1 = d_loss_fn(out_1, ideal_IQ_1)
        d_loss_2 = d_loss_fn(out_2, ideal_IQ_2)
        loss_sup_d = d_loss_0 + d_loss_1 + d_loss_2

        # === 置信度正则 ===
        attconf_reg = attconf_reg_loss(attconf_0) + attconf_reg_loss(attconf_1) + attconf_reg_loss(attconf_2)

        # === inter_graph 稀疏正则 ===
        inter_sparse = inter_sparse_loss(inter_graph_0) + inter_sparse_loss(inter_graph_1) + inter_sparse_loss(inter_graph_2)        
        
        loss_total = loss_sup \
                   + 0.1 * loss_sup_d \
                   + self.lambda_attconf_reg * attconf_reg \
                   + self.lambda_inter_sparse * inter_sparse

        return loss_total


class GLoss_test(nn.Module):
    def __init__(self, weight=0.05):
        super(GLoss_test, self).__init__()
        self.weight = weight
        self.max_range_iq = 0.15
        self.max_range_d = 1

    def forward(self, out_0, out_1, out_2, ideal_IQ, ideal_d):
        """
        :param out: [batch_size, 2, H, W]*3
        :param ideal: [batch_size, 2, H, W]
        :return:
        """
        ideal_IQ_0 = ideal_IQ[:,0:2,:,:]
        ideal_IQ_1 = ideal_IQ[:,2:4,:,:]
        ideal_IQ_2 = ideal_IQ[:,4:6,:,:]

        d_mask = (ideal_d != 0) * (ideal_d < 10) 
        iq_mask = torch.concatenate([d_mask,d_mask], axis=1)

        iq_loss_0 = torch.abs(out_0[iq_mask] - ideal_IQ_0[iq_mask]).mean()
        iq_loss_1 = torch.abs(out_1[iq_mask] - ideal_IQ_1[iq_mask]).mean()
        iq_loss_2 = torch.abs(out_2[iq_mask] - ideal_IQ_2[iq_mask]).mean()
        
        d_0 = iq2d(out_0)
        d_1 = iq2d(out_1)
        d_2 = iq2d(out_2)
        d_ideal_0 = iq2d(ideal_IQ_0)
        d_ideal_1 = iq2d(ideal_IQ_1)
        d_ideal_2 = iq2d(ideal_IQ_2)
        d_loss_0 = torch.abs(d_0[d_mask] - d_ideal_0[d_mask]).mean()
        d_loss_1 = torch.abs(d_1[d_mask] - d_ideal_1[d_mask]).mean()
        d_loss_2 = torch.abs(d_2[d_mask] - d_ideal_2[d_mask]).mean()
        

        """ L1 loss """
        loss_sup = iq_loss_0 + iq_loss_1 + iq_loss_2 
        loss_sup_d = d_loss_0 + d_loss_1 + d_loss_2
        
        return loss_sup+0.1*loss_sup_d


class GLoss_MSE(nn.Module):
    def __init__(self, device, weight=0.05):
        super(GLoss_MSE, self).__init__()
        self.weight = weight
        self.max_range_iq = 0.15
        self.max_range_d = 1
        self.device = device

    def forward(self, out_0, out_1, out_2, ideal_IQ, ideal_d):
        """
        :param out: [batch_size, 2, H, W]*3
        :param ideal: [batch_size, 2, H, W]
        :return:
        """
        ideal_IQ_0 = ideal_IQ[:,0:2,:,:]
        ideal_IQ_1 = ideal_IQ[:,2:4,:,:]
        ideal_IQ_2 = ideal_IQ[:,4:6,:,:]

        d_mask = (ideal_d != 0) * (ideal_d < 10) 
        iq_mask = torch.cat([d_mask,d_mask], axis=1)
        other_d = torch.ones(ideal_d.shape).contiguous().to(self.device) 
        other_iq = torch.ones(ideal_IQ_0.shape).contiguous().to(self.device)

        iq_loss_0 = torch.min((out_0[iq_mask] - ideal_IQ_0[iq_mask]) ** 2, self.max_range_iq ** 2 * other_iq[iq_mask]).mean()
        iq_loss_1 = torch.min((out_1[iq_mask] - ideal_IQ_1[iq_mask]) ** 2, self.max_range_iq ** 2 * other_iq[iq_mask]).mean()
        iq_loss_2 = torch.min((out_2[iq_mask] - ideal_IQ_2[iq_mask]) ** 2, self.max_range_iq ** 2 * other_iq[iq_mask]).mean()
        
        d_0 = iq2d(out_0)
        d_1 = iq2d(out_1)
        d_2 = iq2d(out_2)
        d_ideal_0 = iq2d(ideal_IQ_0)
        d_ideal_1 = iq2d(ideal_IQ_1)
        d_ideal_2 = iq2d(ideal_IQ_2)
        d_loss_0 = torch.min((d_0[d_mask] - d_ideal_0[d_mask]) ** 2, self.max_range_d ** 2 * other_d[d_mask]).mean()
        d_loss_1 = torch.min((d_1[d_mask] - d_ideal_1[d_mask]) ** 2, self.max_range_d ** 2 * other_d[d_mask]).mean()
        d_loss_2 = torch.min((d_2[d_mask] - d_ideal_2[d_mask]) ** 2, self.max_range_d ** 2 * other_d[d_mask]).mean()
        
        """ L1 loss """
        loss_sup = iq_loss_0 + iq_loss_1 + iq_loss_2 
        return loss_sup


class GLoss_MSE_test(nn.Module):
    def __init__(self, weight=0.05):
        super(GLoss_MSE_test, self).__init__()
        self.weight = weight
        self.max_range_iq = 0.15
        self.max_range_d = 1

    def forward(self, out_0, out_1, out_2, ideal_IQ, ideal_d):
        """
        :param out: [batch_size, 2, H, W]*3
        :param ideal: [batch_size, 2, H, W]
        :return:
        """
        ideal_IQ_0 = ideal_IQ[:,0:2,:,:]
        ideal_IQ_1 = ideal_IQ[:,2:4,:,:]
        ideal_IQ_2 = ideal_IQ[:,4:6,:,:]

        d_mask = (ideal_d != 0) * (ideal_d < 10) 
        iq_mask = torch.cat([d_mask,d_mask], axis=1)

        iq_loss_0 = ((out_0[iq_mask] - ideal_IQ_0[iq_mask]) ** 2).mean()
        iq_loss_1 = ((out_1[iq_mask] - ideal_IQ_1[iq_mask]) ** 2).mean()
        iq_loss_2 = ((out_2[iq_mask] - ideal_IQ_2[iq_mask]) ** 2).mean()
        
        d_0 = iq2d(out_0)
        d_1 = iq2d(out_1)
        d_2 = iq2d(out_2)
        d_ideal_0 = iq2d(ideal_IQ_0)
        d_ideal_1 = iq2d(ideal_IQ_1)
        d_ideal_2 = iq2d(ideal_IQ_2)
        d_loss_0 = ((d_0[d_mask] - d_ideal_0[d_mask]) ** 2).mean()
        d_loss_1 = ((d_1[d_mask] - d_ideal_1[d_mask]) ** 2).mean()
        d_loss_2 = ((d_2[d_mask] - d_ideal_2[d_mask]) ** 2).mean()
        

        """ L1 loss """
        loss_sup = iq_loss_0 + iq_loss_1 + iq_loss_2 
        loss_sup_d = d_loss_0 + d_loss_1 + d_loss_2
        
        return loss_sup
    