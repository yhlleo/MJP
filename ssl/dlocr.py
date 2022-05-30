# Based on: 
#   Efficient Training of Visual Transformers with Small Datasets,
#     paper: https://arxiv.org/abs/2106.03746
#     code: https://github.com/yhlleo/VTs-Drloc

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from munch import Munch
from torch import pca_lowrank as PCA

def collect_pos_indexes(mask, normalize=True):
    """
        mask, torch.Tensor [1, C-1], removed the cls 
    """
    mask1d = mask.squeeze(0).cpu().data.numpy() # [C-1] 
    h = w = np.sqrt(mask1d.shape[0])
    nonzero = np.nonzero(mask1d)[0] # -> [num] 
    pts = np.zeros((len(nonzero), 2))
    pts[:, 0], pts[:, 1] = nonzero // h, nonzero % h
    pts = torch.from_numpy(pts).float()
    
    if normalize:
        pts /= h # normalize to [0, 1]
    pts = pts.unsqueeze(0) # [1, num, 2]
    return nonzero, pts

class DenseLocRegress(nn.Module):
    def __init__(self, 
        in_dim, 
        out_dim=2,           # output 2D position
        mlp_type='linear'    # linear, nonlinear, pca
    ):
        super(DenseLocRegress, self).__init__()
        self.in_dim   = in_dim
        self.mlp_type = mlp_type

        self.out_dim = out_dim
        if mlp_type == "nonlinear":
            self.layers = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, self.out_dim)
            )
        elif mlp_type == "linear":
            self.layers = nn.Linear(in_dim, self.out_dim)
        else:
            self.layers = nn.Identity()

    def forward(self, x, mask, normalize=False):
        # x, position embeddings: [1, C, D]
        # mask, torch.Tensor [1, C]
        nonzero, xy = collect_pos_indexes(mask, normalize)
        if isinstance(x, nn.Embedding):
            sample_pos = x(torch.from_numpy(nonzero+1).to(x.weight.device))
            sample_pos = sample_pos.unsqueeze(0)
        else:
            sample_pos = x[:,nonzero+1,:]
        predxy = self.layers(sample_pos)
        
        # use linear PCA projection
        if self.mlp_type == "pca":
            V = PCA(predxy, q=self.out_dim, niter=5)[-1]
            predxy = predxy.bmm(V)

        return xy.to(predxy.device), predxy

