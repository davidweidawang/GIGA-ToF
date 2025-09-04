'''
Author: Jin Zeng, Weida Wang
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
    """3x3 conv with manual zero-padding to keep spatial size."""
    def __init__(self, input_channels, output_channels):
        super(conv3x3, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, 1, 0)
    def forward(self, input):
        input = F.pad(input, (1, 1, 1, 1), mode='constant').contiguous()
        return self.conv(input)
    

class feat_extract_submodule(nn.Module):
    """
    A small stack of 3x3 conv + LeakyReLU blocks.
    Args:
        input_channels: in channels
        output_channels: out channels for each conv
        conv_depth: number of conv layers (>=1)
    """
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
    """
    Final feature block variant:
    - If conv_depth == 1: a single conv (no activation).
    - If conv_depth > 1: (conv + LReLU) * (conv_depth-1) then a final conv (no activation).
    """
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
    """Iterative refinement on I/Q maps using spatially varying convolution (svconv)."""
    def __init__(self, iter_time):
        super(IQRefine, self).__init__()
        self.times = iter_time

    def svconv(self, input, kernel, kernel_size, stride=1, padding=0, dilation=1, native_impl=False):
        """
        Spatially varying convolution implemented via unfold + elementwise weighting.
        Shapes (nominal):
          input:        [B, C, H, W]
          kernel:       [B, 1, K, K, H, W]  (broadcast across C)
        """
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        (bs, ch), in_sz = input.shape[:2], input.shape[2:]

        cols = F.unfold(input, kernel_size, dilation, padding, stride)  # [B, C*K*K, H*W]
        # Reshape to [B, C, K, K, H, W] and apply spatially varying kernel
        output = cols.view(bs, ch, *kernel.shape[2:]) * kernel
        # Sum over K, K, and C -> [B, 1, H, W] squeezed to [B, 1, H, W] per channel-aggregated response
        output = torch.einsum('ijklmn->ijmn', (output,))
        return output

    def forward(self, x_init, affinity_bias, mapped_kernel, mu, attconf, K=3):
        """
        Args:
          x_init:        [B, 1, H, W], the initial I or Q channel
          affinity_bias: [B, K*K+1, H, W], first K*K for weights, last 1 is bias
          mapped_kernel: [B, 9, H, W], the mapped (composed) 3x3 kernel
          mu:            [B, 1, H, W], per-pixel balancing weight
          attconf:       [B, 1, H, W], attention confidence
        """
        B, C, H, W = x_init.size()
        w_pre = 0.1
        x_result = x_init

        # Fuse local affinity and attention-guided kernel
        kernel_sum = F.softmax(affinity_bias[:,:K*K,:,:], dim=1) + attconf*mapped_kernel
        # Normalize along the kernel dimension (ensure positivity and avoid div-by-zero)
        kernel = kernel_sum / kernel_sum.sum(dim=1, keepdim=True).clamp(min=1e-10)
        kernel = kernel.reshape(B,3,3,H,W).unsqueeze(dim=1)  # [B, 1, 3, 3, H, W]

        bias = affinity_bias[:,K*K:,:,:]  # [B, 1, H, W]

        # Fixed-point style iterative update
        for i in range(self.times):
            # Data + prior blending
            x_result = (mu * x_result +  w_pre*x_init)/(mu + w_pre)
            # Spatially varying convolution
            x_result = self.svconv(x_result, kernel, kernel_size=3, stride=1, padding=1, dilation=1)
            
        return x_result + bias

class GIGA(nn.Module):
    def __init__(self, dim=64, num_heads=1, qkv_bias=True, window_size=(7,7), attn_drop=0., proj_drop=0.):
        """
        GIGA: Graph-Informed Geometric Attention (local-window correlation).
        Args:
          dim:        channel dimension of input features
          num_heads:  number of heads (dim must be divisible by num_heads)
          qkv_bias:   whether to use bias in linear layers
          attn_drop:  dropout for attention weights
          proj_drop:  reserved (not used here)
        """
        super(GIGA, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.dim_qk = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        
        self.scale = self.dim_qk ** -0.5
        self.q = nn.Linear(dim, self.dim_qk, bias=qkv_bias)
        self.k = nn.Linear(dim, self.dim_qk, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=1)

        # Confidence branch over a 7x7 (49) local window
        self.attn_conf = nn.Linear(49, 1, bias=qkv_bias)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization
        nn.init.trunc_normal_(self.q.weight, std=0.02)
        nn.init.trunc_normal_(self.k.weight, std=0.02)
        nn.init.trunc_normal_(self.attn_conf.weight, std=0.02)
        if self.q.bias is not None:
            nn.init.constant_(self.q.bias, 0)
            nn.init.constant_(self.k.bias, 0)
            nn.init.constant_(self.attn_conf.bias, 0)

    def forward(self, x, x_pre, mask=None):
        """
        Cross-frame attention between current x and previous x_pre at 1/8 scale.
        Returns:
          attn:    [B, 49, H, W] softmaxed local attention across a 7x7 window
          attconf: [B, 1,  H, W] confidence map in [0,1]
        """
        B, C, H, W = x.size()  # x_pre has the same shape, features at h/8, w/8
        
        q = self.q(x.permute(0,2,3,1)).permute(0,3,1,2)      # [B, C, H, W]
        k = self.k(x_pre.permute(0,2,3,1)).permute(0,3,1,2)  # [B, C, H, W]
        
        window = 7        
        window_size = (window, window)
        dilation = (1, 1)
        stride = (1, 1)
        padding = (3, 3)  # for 7x7 window with dilation=1
        cols = F.unfold(k, _pair(window_size), _pair(dilation), _pair(padding), _pair(stride)) # [B, C*49, H*W]

        # Reshape to [B, C, 49, H, W] and correlate with q
        attn = cols.view(B, C, window*window, H, W) * q.unsqueeze(dim=2)   # [B, C, 49, H, W]
        # Sum over channel dimension and scale -> [B, 49, H, W]
        attn = (torch.einsum('ijklm->iklm', (attn,))) * self.scale

        # Predict attention confidence per location: [B, 1, H, W]
        attconf = self.attn_conf(attn.permute(0,2,3,1)).permute(0,3,1,2)
            
        # Softmax over the 49 local positions and dropout
        attn = attn.softmax(dim=1)        
        attn = self.attn_drop(attn)
        return attn, torch.sigmoid(attconf)
    
    
class GIGAToF(nn.Module):
    """
    U-Net style backbone + GIGA cross-frame attention + iterative IQ refinement.
    """
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

        # Encoder / decoder scaffolding
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.up = nn.ModuleList()
        self.bilinear = nn.ModuleList()

        # First down block at full scale
        self.down.append(feat_extract_submodule(input_channel, curr_chn[0], conv_depth = conv_depth))

        # Downsampling path
        for i in range(unet_depth):
            self.pool.append(nn.AvgPool2d(2))
            self.down.append(feat_extract_submodule(curr_chn[i], curr_chn[i + 1], conv_depth = conv_depth))

        # Upsampling path (all but last)
        for i in range(unet_depth - 2):
            self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.up.append(feat_extract_submodule(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], curr_chn[unet_depth + i + 1], conv_depth = conv_depth))
        
        # Final up block
        i = unet_depth - 2
        self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))
        self.up.append(feat_extract_submodule_fin(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], curr_chn[unet_depth + i + 1], conv_depth = conv_depth))

        # One more upsample for guide maps
        self.bilinear.append(nn.UpsamplingBilinear2d(scale_factor=2))

        # Mu decoding head at the penultimate concatenation level
        self.mu_dec0_1 = feat_extract_submodule_fin(curr_chn[unet_depth + i] + curr_chn[unet_depth - i - 1], input_channel, conv_depth = conv_depth)

        # GIGA module:
        # - pre_graph produces two 3x3 (K*K) intra-graphs from the previous frame (I and Q)
        self.pre_graph = feat_extract_submodule_fin(curr_chn[unet_depth], input_channel*K*K, conv_depth = conv_depth)
        self.GIGA = GIGA()  # inter-graph over a 7x7 window

        # IQ iterative refinement
        self.iq_refine = IQRefine(5)
    
    def mapped_graph(self, A_local_1, A_local_2, neighborhood_size_1, neighborhood_size_2):
        """
        Convert local adjacency A_local_* (B x H x W x N) into a dense (B x HW x HW),
        compose them, and re-sample back to 3x3 neighborhoods.
        Assumptions:
          - A_local_1 corresponds to a neighborhood_size_1^2 window (e.g., 7x7 => 49).
          - A_local_2 corresponds to a neighborhood_size_2^2 window (e.g., 3x3 => 9).
        Returns:
          A_bhw9: [B, H, W, 9] composed adjacency restricted to 3x3 neighbors.
        """

        B, H, W, adj_size_1 = A_local_1.shape
        _, _, _, adj_size_2 = A_local_2.shape
        num_nodes = H * W
        
        # Flatten spatial indices per node
        row_grid = torch.arange(H).repeat_interleave(W).view(-1, 1).to(A_local_1.device)  # [H*W, 1]
        col_grid = torch.arange(W).repeat(H).view(-1, 1).to(A_local_1.device)             # [H*W, 1]

        # Neighbor offsets (assume square windows)
        # For A_local_1
        offsets_1 = torch.arange(-neighborhood_size_1//2 + 1 , neighborhood_size_1//2 + 1, device=A_local_1.device)
        grid_y_1, grid_x_1 = torch.meshgrid(offsets_1, offsets_1, indexing='ij')
        grid_y_1 = grid_y_1.reshape(-1)  # (adj_size_1,)
        grid_x_1 = grid_x_1.reshape(-1)  # (adj_size_1,)
        
        # For A_local_2 (e.g., 3x3)
        offsets_2 = torch.arange(-neighborhood_size_2//2 + 1 , neighborhood_size_2//2 + 1, device=A_local_2.device)
        grid_y_2, grid_x_2 = torch.meshgrid(offsets_2, offsets_2, indexing='ij')
        grid_y_2 = grid_y_2.reshape(-1)  # (adj_size_2,)
        grid_x_2 = grid_x_2.reshape(-1)  # (adj_size_2,)

        # Neighbor indices for A_local_1
        neighbor_rows_1 = (row_grid + grid_y_1.view(1, -1)).clamp(0, H - 1)  # (num_nodes, adj_size_1)
        neighbor_cols_1 = (col_grid + grid_x_1.view(1, -1)).clamp(0, W - 1)  # (num_nodes, adj_size_1)
        col_indices_1 = neighbor_rows_1 * W + neighbor_cols_1                 # (num_nodes, adj_size_1)

        # Neighbor indices for A_local_2
        neighbor_rows_2 = (row_grid + grid_y_2.view(1, -1)).clamp(0, H - 1)  # (num_nodes, adj_size_2)
        neighbor_cols_2 = (col_grid + grid_x_2.view(1, -1)).clamp(0, W - 1)  # (num_nodes, adj_size_2)
        col_indices_2 = neighbor_rows_2 * W + neighbor_cols_2                 # (num_nodes, adj_size_2)

        # Expand for batch
        values_1 = A_local_1.reshape(B, num_nodes, adj_size_1)
        values_2 = A_local_2.reshape(B, num_nodes, adj_size_2)

        # Dense adjacency matrices for A_local_1 and A_local_2
        adj_matrix_1 = torch.zeros((B, num_nodes, num_nodes), device=A_local_1.device)
        col_indices_1_exp = col_indices_1.unsqueeze(0).expand(B, -1, -1)
        adj_matrix_1.scatter_(2, col_indices_1_exp, values_1)

        adj_matrix_2 = torch.zeros((B, num_nodes, num_nodes), device=A_local_2.device)
        col_indices_2_exp = col_indices_2.unsqueeze(0).expand(B, -1, -1)
        adj_matrix_2.scatter_(2, col_indices_2_exp, values_2)

        # Compose: A1 * A2 * A1^T
        adj_matrix = torch.bmm(torch.bmm(adj_matrix_1, adj_matrix_2), adj_matrix_1.transpose(-2, -1))

        # Keep only edges within the A_local_2 neighborhood
        A_sparse_3x3 = adj_matrix * (adj_matrix_2 > 0).float()

        # Convert 2D neighbor coords to flattened 1D indices for A_local_2 (assumed to be 3x3 -> 9)
        neighbor_indices = (neighbor_rows_2 * W + neighbor_cols_2).view(num_nodes, 9)  # (num_nodes, 9)

        # Gather values and reshape to [B, H, W, 9]
        A_bhw9 = A_sparse_3x3[:, torch.arange(num_nodes).unsqueeze(1), neighbor_indices]  # (B, num_nodes, 9)
        A_bhw9 = A_bhw9.view(B, H, W, 9)

        return A_bhw9
    
    def forward(self, concat_IQ, concat_IQ_pre):
        """
        Args:
          concat_IQ:     [B, 2, H, W] current frame (I and Q)
          concat_IQ_pre: [B, 2, H, W] previous frame (I and Q)

        Returns:
          xout:      [B, 2, H, W] refined I and Q
          mu:        [B, 2, H, W] per-channel mu map
          inter_graph: [B, 49, H/8, W/8] inter-frame attention
          attconf:     [B, 1,  H/8, W/8] attention confidence
        """
        unet_depth = self.unet_depth
        feature_maps = []
        feature_maps_pre = []
        
        # Current frame: encode down to 1/8 scale
        feature_maps.append(self.down[0](concat_IQ))
        for i in range(unet_depth):
            pooled = self.pool[i](feature_maps[i])
            feature_maps.append(self.down[i+1](pooled))

        # Decode up to 1/2 scale (keep last concatenation tensor)
        for i in range(unet_depth - 1):
            bilineared = self.bilinear[i](feature_maps[unet_depth + i])
            concated = torch.cat((bilineared, feature_maps[unet_depth - i - 1]), 1)
            feature_maps.append(self.up[i](concated))
        
        # Guide features: upsample to full scale
        guide = self.bilinear[unet_depth - 1](feature_maps[-1])

        # Previous frame: encode down to 1/8 scale (deepest feature)
        feature_maps_pre.append(self.down[0](concat_IQ_pre))
        for i in range(unet_depth):
            pooled = self.pool[i](feature_maps_pre[i])
            feature_maps_pre.append(self.down[i+1](pooled)) 
            
        # Intra-graph (previous frame): two KxK (3x3) kernels for I and Q
        guide_pre = self.pre_graph(feature_maps_pre[-1])  # [B, 2*(9+1), H/8, W/8]
        pre_intra_0 = F.softmax(guide_pre[:,:(self.K*self.K),:,:], dim=1)                # [B, 9, H/8, W/8]
        pre_intra_1 = F.softmax(guide_pre[:,(self.K*self.K):(2*self.K*self.K),:,:], dim=1)  # [B, 9, H/8, W/8]
        
        # Inter-graph between current and previous
        inter_graph, attconf = self.GIGA(feature_maps[unet_depth], feature_maps_pre[unet_depth])  # [B,49,H/8,W/8]
        attconf_return = attconf
        
        # Add self-loop at the center of the 7x7 window (index 24)
        selfloop = torch.zeros_like(inter_graph)
        selfloop[:, 24, :, :] = 1      

        # Compose adjacency and remap to 3x3 for I and Q branches
        mapped_graph_0 = self.mapped_graph((inter_graph+selfloop).permute(0,2,3,1), pre_intra_0.permute(0,2,3,1), 7, 3).permute(0,3,1,2)  # [B, 9, H/8, W/8]
        mapped_graph_1 = self.mapped_graph((inter_graph+selfloop).permute(0,2,3,1), pre_intra_1.permute(0,2,3,1), 7, 3).permute(0,3,1,2)

        # Upsample mapped 3x3 kernels and attconf back to full resolution
        for i in range(unet_depth):
            mapped_graph_0 = self.bilinear[i](mapped_graph_0)
            mapped_graph_1 = self.bilinear[i](mapped_graph_1)
            attconf = self.bilinear[i](attconf)
        
        # Mu decoding
        mu_feat = self.mu_dec0_1(concated)                     # at the last concat scale
        mu = self.bilinear[unet_depth - 1](mu_feat)            # upsample to full res
        mu = torch.sigmoid(mu)

        # Ablation path (as in original): try attconf=1, then regularize, then sparsify inter_graph, etc.
        xout_0 = self.iq_refine(concat_IQ[:,0:1,:,:], guide[:,:(self.K*self.K+1),:,:], mapped_graph_0, mu[:,0:1,:,:], attconf)
        xout_1 = self.iq_refine(concat_IQ[:,1:2,:,:], guide[:,(self.K*self.K+1):,:,:], mapped_graph_1, mu[:,1:2,:,:], attconf)

        return torch.concat((xout_0, xout_1),axis=1), mu, inter_graph, attconf_return



if __name__ == "__main__":
    # Demo inputs
    batch_size = 1
    channels = 2
    height = 320
    width = 240

    concat_IQ = torch.rand((batch_size, channels, height, width))
    concat_IQ_pre = torch.rand((batch_size, channels, height, width))

    model = GIGAToF()

    # Forward returns four tensors; unpack accordingly
    output, mu, inter_graph, attconf = model(concat_IQ, concat_IQ_pre)

    print("Output shape:", output.shape)
    print("Mu shape:", mu.shape)