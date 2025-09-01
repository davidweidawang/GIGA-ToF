'''
Author: Jin Zeng
Date: 2025-01-20
Description: Graph-Informed Geometric Attention for temporal iToF denoising
'''
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .common import *
from torch.utils.data import Dataset
from torchvision.utils import save_image


class conv3x3(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, 1, 0)
    def forward(self, input):
        input = F.pad(input, (1, 1, 1, 1), mode='constant').contiguous()
        return self.conv(input)
    

class feat_extract_submodule(nn.Module):
    def __init__(self, input_channels, output_channels, conv_depth = 3):
        super(feat_extract_submodule, self).__init__()
        submodule = []
        submodule.append(conv3x3(input_channels, output_channels))
        submodule.append(nn.LeakyReLU())
        for i in range(conv_depth - 1):
            submodule.append(conv3x3(output_channels, output_channels))
            submodule.append(nn.LeakyReLU())
        self.seq = nn.Sequential(*submodule)
    def forward(self, input):
        return self.seq(input)


class feat_extract_submodule_fin(nn.Module):
    def __init__(self, input_channels, output_channels, conv_depth = 3):
        super(feat_extract_submodule_fin, self).__init__()
        submodule = []
        submodule.append(conv3x3(input_channels, output_channels))
        if conv_depth>1:
            submodule.append(nn.LeakyReLU())
            for i in range(conv_depth - 2):
                submodule.append(conv3x3(output_channels, output_channels))
                submodule.append(nn.LeakyReLU())
            submodule.append(conv3x3(output_channels, output_channels))
        
        self.seq = nn.Sequential(*submodule)
    def forward(self, input):
        return self.seq(input)
    

class IQRefine(nn.Module):
    def __init__(self, iter_time):
        super(IQRefine, self).__init__()
        self.times = iter_time

    def svconv(self, input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        (bs, ch), in_sz = input.shape[:2], input.shape[2:]

        cols = F.unfold(input, kernel_size, dilation, padding, stride)
        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        output = torch.einsum('ijklmn->ijmn', (output,))
        return output

    def forward(self, x_init, affinity_bias, mapped_kernel, mu, attconf, K=3):
        B, C, H, W = x_init.size()
        w_pre = 0.1
        x_result = x_init

        kernel_sum = F.softmax(affinity_bias[:,:K*K,:,:], dim=1) + attconf*mapped_kernel
        # Normalizes non-negative kernel so that dim 1 sums to 1
        kernel = kernel_sum / kernel_sum.sum(dim=1, keepdim=True).clamp(min=1e-10)  # Avoid division by zero
        # norm with sum
        
        kernel = kernel.reshape(B,3,3,H,W).unsqueeze(dim=1)

        bias = affinity_bias[:,K*K:,:,:]

        for i in range(self.times):
            x_result = (mu * x_result +  w_pre*x_init)/(mu + w_pre)   # for ablation 2
            x_result = self.svconv(x_result, kernel, kernel_size=3, stride=1, padding=1, dilation=1)           
            
        return x_result+bias
    

class GIGA(nn.Module):
    def __init__(self, dim=64, num_heads=1, qkv_bias=True, window_size=(7,7), attn_drop=0., proj_drop=0.):
        """
        参数说明：
          dim: 输入特征的通道数（本例为64）
          num_heads: 注意力头的数量 num_heads=4
          qkv_bias: 是否在线性变换中使用偏置
          attn_drop, proj_drop: dropout概率
        """
        super(GIGA, self).__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.dim = dim
        self.dim_qk = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        
        self.scale = self.dim_qk ** -0.5
        self.q = nn.Linear(dim, self.dim_qk, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim_qk, bias=qkv_bias)

        # 输出投影层
        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=1)

        # 输出confidence层
        self.attn_conf = nn.Linear(49, 1, bias=qkv_bias)

        self._init_weights()

    def _init_weights(self):
        # 使用截断正态初始化
        nn.init.trunc_normal_(self.q.weight, std=0.02)
        nn.init.trunc_normal_(self.k.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_conf.weight, std=0.02)
        # nn.init.trunc_normal_(self.proj.weight, std=0.02)
        if self.q.bias is not None:
            nn.init.constant_(self.q.bias, 0)
            nn.init.constant_(self.k.bias, 0)
            nn.init.constant_(self.attn_conf.bias, 0)

    def forward(self, x, x_pre, mask=None):
        B, C, H, W = x.size() # x_pre is the same, [batch, C, h/8, w/8]
        
        # 可以先去掉检查代码逻辑
        q = self.q(x.permute(0,2,3,1)).permute(0,3,1,2) # [b c h w], c=dim/2
        k = self.k(x_pre.permute(0,2,3,1)).permute(0,3,1,2) # [b c h w], c=dim/2
        
        window = 7        
        window_size = (window, window)
        dilation = (1, 1)
        stride = (1, 1)
        padding = (3, 3)  # padding 通常取 (kernel_size - 1) // 2，当 dilation=1 时
        cols = F.unfold(k, _pair(window_size), _pair(dilation), _pair(padding), _pair(stride)) # [b,c*49,h*w]

        attn = cols.view(B,C,window*window,H,W) * q.unsqueeze(dim=2)    #[b,c,49,h,w]
        attn = (torch.einsum('ijklm->iklm', (attn,))) * self.scale # [b,c,49,h,w] -> [b,49,h,w]

        attconf = self.attn_conf(attn.permute(0,2,3,1)).permute(0,3,1,2)
            
        attn = attn.softmax(dim=1)        
        attn = self.attn_drop(attn) # [b,49,h,w]
        return attn, torch.sigmoid(attconf)
    
    
class GIGAToF(nn.Module):
    def __init__(self, input_channel=2, ch0 = 16, K = 3, conv_depth = 2, unet_depth = 3):
        super(GIGAToF, self).__init__()
        self.K = K
        self.channels = [ch0]
        self.unet_depth = unet_depth
        for i in range(unet_depth - 1):
            self.channels.append(ch0 * 2)
            ch0 *= 2
        self.channels.append(ch0)
        if unet_depth>2:
            self.channels.append(ch0)
            for i in range(unet_depth - 3):
                self.channels.append(ch0 // 2)
                ch0 //= 2
        self.channels.append(input_channel*K*K+2)
        curr_chn = self.channels
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.up = nn.ModuleList()
        self.bilinear = nn.ModuleList()
        self.down.append(feat_extract_submodule(input_channel, curr_chn[0], conv_depth = conv_depth))
        for i in range(unet_depth):
            self.pool.append(nn.AvgPool2d(2))
            self.down.append(feat_extract_submodule(curr_chn[i], curr_chn[i + 1], conv_depth = conv_depth))
        for i in range(unet_depth - 2):
            self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.up.append(feat_extract_submodule(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], curr_chn[unet_depth + i + 1], conv_depth = conv_depth))
        
        i = unet_depth - 2
        self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.up.append(feat_extract_submodule_fin(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], curr_chn[unet_depth + i + 1], conv_depth = conv_depth))

        self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))

        self.mu_dec0_1 = feat_extract_submodule_fin(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], input_channel, conv_depth = conv_depth)

        # GIGA-module
        self.pre_graph = feat_extract_submodule_fin(curr_chn[unet_depth], input_channel*K*K, conv_depth = conv_depth)
        self.GIGA = GIGA() # inter-graph

        # Depth refinement
        self.iq_refine = IQRefine(5)
    
    def mapped_graph(self, A_local_1, A_local_2, neighborhood_size_1, neighborhood_size_2):
        """Convert local adjacency matrix (B x H x W x N) into a dense adjacency matrix (B x HW x HW)."""

        B, H, W, adj_size_1 = A_local_1.shape  # Batch size, Height, Width, Number of neighbors
        _, _, _, adj_size_2 = A_local_2.shape  # Batch size, Height, Width, Number of neighbors 
        num_nodes = H * W  # Total nodes per graph
        
        # Flatten spatial indices
        row_grid = torch.arange(H).repeat_interleave(W).view(-1, 1).to(A_local_1.device)  # Row index per node [H*W,1]
        col_grid = torch.arange(W).repeat(H).view(-1, 1).to(A_local_1.device)  # Column index per node [H*W,1]

        # Compute neighbor offsets; 这里假设邻域为平方区域，adj_size_1 应该等于 neighborhood_size_1^2
        # 对 A_local_1
        offsets_1 = torch.arange(-neighborhood_size_1//2 + 1 , neighborhood_size_1//2 + 1, device=A_local_1.device)
        # 生成二维偏移 (假设邻域为正方形)，这里需要生成所有组合
        grid_y_1, grid_x_1 = torch.meshgrid(offsets_1, offsets_1, indexing='ij')
        grid_y_1 = grid_y_1.reshape(-1)  # shape: (adj_size_1,)
        grid_x_1 = grid_x_1.reshape(-1)  # shape: (adj_size_1,)
        
        # 对 A_local_2，类似地，假设邻域大小为 neighborhood_size_2^2 (比如 3x3)
        offsets_2 = torch.arange(-neighborhood_size_2//2 + 1 , neighborhood_size_2//2 + 1, device=A_local_2.device)
        grid_y_2, grid_x_2 = torch.meshgrid(offsets_2, offsets_2, indexing='ij')
        grid_y_2 = grid_y_2.reshape(-1)  # shape: (adj_size_2,)
        grid_x_2 = grid_x_2.reshape(-1)  # shape: (adj_size_2,)

        # Compute neighbor indices for A_local_1
        neighbor_rows_1 = (row_grid + grid_y_1.view(1, -1)).clamp(0, H - 1)  # (num_nodes, adj_size_1)
        neighbor_cols_1 = (col_grid + grid_x_1.view(1, -1)).clamp(0, W - 1)  # (num_nodes, adj_size_1)
        col_indices_1 = neighbor_rows_1 * W + neighbor_cols_1  # (num_nodes, adj_size_1)

        # Compute neighbor indices for A_local_2
        neighbor_rows_2 = (row_grid + grid_y_2.view(1, -1)).clamp(0, H - 1)  # (num_nodes, adj_size_2)
        neighbor_cols_2 = (col_grid + grid_x_2.view(1, -1)).clamp(0, W - 1)  # (num_nodes, adj_size_2)
        col_indices_2 = neighbor_rows_2 * W + neighbor_cols_2  # (num_nodes, adj_size_2)

        # Expand for batch processing
        values_1 = A_local_1.reshape(B, num_nodes, adj_size_1)
        values_2 = A_local_2.reshape(B, num_nodes, adj_size_2)

        # Construct dense adjacency matrices for A_local_1 and A_local_2
        adj_matrix_1 = torch.zeros((B, num_nodes, num_nodes), device=A_local_1.device)
        # 扩展 col_indices_1 到 batch 维度： (B, num_nodes, adj_size_1)
        col_indices_1_exp = col_indices_1.unsqueeze(0).expand(B, -1, -1)
        # Scatter values into adj_matrix_1 沿着 dim=2
        adj_matrix_1.scatter_(2, col_indices_1_exp, values_1)

        adj_matrix_2 = torch.zeros((B, num_nodes, num_nodes), device=A_local_2.device)
        col_indices_2_exp = col_indices_2.unsqueeze(0).expand(B, -1, -1)
        adj_matrix_2.scatter_(2, col_indices_2_exp, values_2)

        # mapped_graph: 进行两次批量矩阵乘法，构造新的邻接矩阵
        adj_matrix = torch.bmm(torch.bmm(adj_matrix_1, adj_matrix_2), adj_matrix_1.transpose(-2, -1))

        # Apply mask to retain only edges in the A_local_2 neighborhood
        A_sparse_3x3 = adj_matrix * (adj_matrix_2 > 0).float()

        # Convert 2D neighbor coordinates to flattened 1D indices for A_local_2 (assumed to be 3x3)
        # 注意：这里硬编码为9个邻居，确保 adj_size_2 实际为 9
        neighbor_indices = (neighbor_rows_2 * W + neighbor_cols_2).view(num_nodes, 9)  # (num_nodes, 9)

        # Gather the relevant values from A_sparse_3x3
        A_bhw9 = A_sparse_3x3[:, torch.arange(num_nodes).unsqueeze(1), neighbor_indices]  # (B, num_nodes, 9)
        # Reshape to (B, H, W, 9)
        A_bhw9 = A_bhw9.view(B, H, W, 9)
        # print(A_bhw9.shape)

        return A_bhw9
    
    def forward(self, concat_IQ, concat_IQ_pre):
        unet_depth = self.unet_depth
        feature_maps = []
        feature_maps_pre = []
        
        # current frame, get f at 1/2 scale
        feature_maps.append(self.down[0](concat_IQ))
        for i in range(unet_depth):
            pooled = self.pool[i](feature_maps[i])
            feature_maps.append(self.down[i+1](pooled))
        for i in range(unet_depth - 1):
            bilineared = self.bilinear[i](feature_maps[unet_depth + i])
            concated = torch.cat((bilineared, feature_maps[unet_depth - i - 1]), 1)
            feature_maps.append(self.up[i](concated))
        
        guide = self.bilinear[unet_depth - 1](feature_maps[-1]) # current frame use 1/2 size and upsample
               
        # previous frame, get f at 1/8 scale (deepest)
        feature_maps_pre.append(self.down[0](concat_IQ_pre))
        for i in range(unet_depth):
            pooled = self.pool[i](feature_maps_pre[i])
            feature_maps_pre.append(self.down[i+1](pooled)) 
            
        guide_pre = self.pre_graph(feature_maps_pre[-1]) # [b 2*(9+1) h/8 w/8] pre-frame intra graph
        pre_intra_0 = F.softmax(guide_pre[:,:(self.K*self.K),:,:], dim=1) # [b 9 h/8 w/8] 
        pre_intra_1 = F.softmax(guide_pre[:,(self.K*self.K):(2*self.K*self.K),:,:], dim=1)  # [b 9 h/8 w/8]
        
        inter_graph, attconf = self.GIGA(feature_maps[unet_depth], feature_maps_pre[unet_depth]) #[b,49,h/8,w/8]   
        attconf_return = attconf
        
        selfloop = torch.zeros_like(inter_graph)
        selfloop[:, 24, :, :] = 1      

        # Compute composite adjacency matrix
        mapped_graph_0 = self.mapped_graph((inter_graph+selfloop).permute(0,2,3,1), pre_intra_0.permute(0,2,3,1), 7, 3).permute(0,3,1,2) #[B, 9, H, W]
        mapped_graph_1 = self.mapped_graph((inter_graph+selfloop).permute(0,2,3,1), pre_intra_1.permute(0,2,3,1), 7, 3).permute(0,3,1,2)

        for i in range(unet_depth):
            mapped_graph_0 = self.bilinear[i](mapped_graph_0) # scale=1      
            mapped_graph_1 = self.bilinear[i](mapped_graph_1) # scale=1   
            attconf = self.bilinear[i](attconf) # scale=1                 
        
        # Mu Decoding
        mu_feat = self.mu_dec0_1(concated) #up
        mu = self.bilinear[unet_depth - 1](mu_feat) #up
        mu = torch.sigmoid(mu)
        # 先试一下attconf为1，再试attconf正则化，再到inter_gragh稀疏
        xout_0 = self.iq_refine(concat_IQ[:,0:1,:,:], guide[:,:(self.K*self.K+1),:,:], mapped_graph_0, mu[:,0:1,:,:], attconf)
        xout_1 = self.iq_refine(concat_IQ[:,1:2,:,:], guide[:,(self.K*self.K+1):,:,:], mapped_graph_1, mu[:,1:2,:,:], attconf)

        return torch.concat((xout_0, xout_1),axis=1), mu, inter_graph, attconf


if __name__ == "__main__":
    # 设置输入参数
    batch_size = 1
    channels = 2
    height = 320
    width = 240

    # 初始化随机输入
    concat_IQ = torch.rand((batch_size, channels, height, width))
    concat_IQ_pre = torch.rand((batch_size, channels, height, width))

    # 初始化模型
    model = GIGAToF()

    # 运行前向传播
    output, mu = model(concat_IQ, concat_IQ_pre)

    # 输出结果形状
    print("Output shape:", output.shape)
    print("Mu shape:", mu.shape)