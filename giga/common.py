'''
Author: Qingpeng Zhu
Email: qzhuad@connect.ust.hk
Date: 2022-07-22 15:37:29
LastEditTime: 2025-01-12 14:57:54
Description: RGBD deep fusion for depth completion
'''

"""
    Non-Local Spatial Propagation Network for Depth Completion
    Jinsun Park, Kyungdon Joo, Zhe Hu, Chi-Kuei Liu and In So Kweon
    European Conference on Computer Vision (ECCV), Aug 2020
    Project Page : https://github.com/zzangjinsun/NLSPN_ECCV20
    Author : Jinsun Park (zzangjinsun@kaist.ac.kr)
    ======================================================================
    Some of useful functions are defined here.
"""


import torch
import torch.nn as nn
import torchvision

model_path = {
    'resnet18': './resnet18.pth',
    #'resnet34': '../pretrained/resnet34.pth'
}

def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained:
        #state_dict = torch.load(model_path['resnet18'])
        state_dict = torch.load('./deepglr/resnet18.pth')
        net.load_state_dict(state_dict)

    return net


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=1, bn=True, relu=True):
    assert (kernel % 2) == 1, 'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0, bn=True, relu=True):
    assert (kernel % 2) == 1, 'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))

    layers = nn.Sequential(*layers)

    return layers