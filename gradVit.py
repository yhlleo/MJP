import numpy as np
import random
import argparse
import torch
import pdb
from asyncio import constants
import os

import cv2
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from timm import create_model
from models.vit_timm import vit_small_patch16_224
from gradVit.gradVit_dataset import GradVitDataset


def image_prior(x, model):
    pass

def patch_prior(x, model):
    pass

def grad_prior(x, model):
    pass

def train(data_path, model=None):
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # --- prepare data ---
    dataset_train = GradVitDataset(
        root_path = data_path
    )
    data = DataLoader(
        dataset_train,
        batch_size=1,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        shuffle=False
    )

    # --- gt_prior ---
    img_prior_gt = image_prior(inp_gt, net)
    patch_prior_gt = patch_prior(inp_gt, net)
    grad_prior_gt = grad_prior(inp_gt, net)

    # --- model ---
    net = model

    # ---loss item & optimizer ---
    loss_mae = nn.L1Loss(reduction = 'mean')

    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=0.1
    )

    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=120000, 
        eta_min=0
    )
    step = 0

    while step <= 120000:
        for img_gt in data:
            inp_gt = img_gt
            pdb.set_trace()
            inp_noise = torch.rand(inp_gt.size()).requires_grad(True)

            # Q: 这个模型的输出需要关注或者监督吗？
            oup_noise = net(inp_noise)

            img_prior_inp = image_prior(inp_noise, net)
            patch_prior_inp = patch_prior(inp_noise, net)
            grad_prior_inp = patch_prior(inp_noise, net)

            loss_total = (loss_mae(img_prior_gt, img_prior_inp) + 
                        loss_mae(patch_prior_gt, patch_prior_inp) + 
                        loss_mae(grad_prior_gt, grad_prior_inp))

            optimizer.zerp_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            step += 1

        if step % 1000 == 1:
            inp_noise.permute(1, 2, 0)
    

if __name__ == "__main__":

    DATA_PATH = "./gradVit/data4GradVit"
    model_type = "vit-small" # mjp, vit-small
    model_path = ""

    # --- model ---
    if model_type == "mjp":
        net = vit_small_patch16_224(
            pretrained=False, 
            num_classes=1000,
            use_unk=True,
            use_idx_emb=False,
            use_dlocr=True,
            dlocr_type='nonlinear',  # nonlinear, linear, pca
            use_avg=False
        ).cuda()
        checkpoint = torch.load(model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model'], strict=True)
    else:
        net = create_model("vit_small_patch16_224", pretrained=True).cuda()

    train(DATA_PATH, model=net)

