from munch import Munch

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON=1e-8

def relative_constraint_l1(xy, predxy):
    return F.l1_loss(xy, predxy)

def cal_selfsupervised_loss(outs, args, lambda_dlocr=0.0):
    loss, all_losses = 0.0, Munch()
    if args.TRAIN.USE_DLOCR:
        criterion = relative_constraint_l1

        loss_dlocr = 0.0
        for xy, predxy in zip(outs.xy, outs.predxy):
            loss_dlocr += criterion(xy, predxy) * lambda_dlocr
        all_losses.dlocr = loss_dlocr.item()
        loss += loss_dlocr

    return loss, all_losses

