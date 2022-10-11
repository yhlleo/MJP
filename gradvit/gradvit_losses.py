import cv2
import torch
import torch.nn as nn


class GradLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, model):
        
        out = inp
        return out 

class ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, model):
        
        out = inp
        return out 

class AuxLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, model):
        
        out = inp
        return out 