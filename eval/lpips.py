"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""
import os
import torch
import torch.nn as nn
from torchvision import models


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=False).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)

class LPIPS(nn.Module):
    def __init__(self, device):
        super(LPIPS, self).__init__()
        self.alexnet = AlexNet().eval()
        self._load_alexnet_weights() # load from download model

        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).to(device)
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).to(device)

        for param in self.alexnet.parameters():
            param.requires_grad = False
        for param in self.lpips_weights.parameters():
            param.requires_grad = False

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load('eval/lpips_weights.ckpt')
        else:
            state_dict = torch.load('eval/lpips_weights.ckpt',
                                    map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap)**2))
        return lpips_value

    def _load_alexnet_weights(self):
        path = "/apdcephfs/share_916081/yliu/pretrained_models/alexnet-owt-4df8aa71.pth"
        state_dict = torch.load(path, map_location=torch.device('cpu'))

        alexnet_state_dict = self.alexnet.layers.state_dict()
        for name, param in state_dict.items():
            if name.replace("features.", "") in alexnet_state_dict:
                alexnet_state_dict[name.replace("features.", "")].copy_(param)

