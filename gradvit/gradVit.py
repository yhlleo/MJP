import numpy as np
import random
import torch
import pdb
import os
import wandb

import cv2
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from timm import create_model
from models.vit_timm import vit_small_patch16_224
from gradvit_dataset import GradVitDataset
from gradvit_losses import image_loss
# from gradvit.gradvit_losses import GradLoss, ImageLoss, AuxLoss

# --- setup environment ---
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

def runGradVit(model=None, data=None, loss_match=None, grad_prior=None, image_prior=None, aux_regular=None):
    
    wandb.init(project='MJP', entity="renbin", name="gradVit_attact")
    
    for img_gt in data:
        # --- input noise ---
        inp_noise = torch.rand(img_gt.size(), requires_grad=True)

        # --- optimizer & lr_scheduler ---
        optimizer = torch.optim.AdamW(
            [inp_noise],
            lr=0.1
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=120000, 
            eta_min=0
        )

        # --- gt priors ---
        grad_prior_gt = grad_prior(img_gt, model) if grad_prior is not None else 0.
        image_prior_gt = image_prior(img_gt, inp_noise) if image_prior is not None else 0.
        aux_regular_gt = aux_regular(img_gt, model) if aux_regular is not None else 0.

        # --- GradVit attact ---
        step = 0
        while step <= 120000:
            # Q: 这个模型的输出需要关注或者监督吗？
            oup_noise = model(inp_noise)

            # --- inp_noise priors
            grad_prior_noise = grad_prior(inp_noise, model) if grad_prior is not None else 0.
            image_prior_noise = image_prior(img_gt, inp_noise) if image_prior is not None else 0.
            aux_regular_noise = aux_regular(inp_noise, model) if aux_regular is not None else 0.

            output = {"grad_prior_noise": grad_prior_noise}
            output["image_prior_noise"] = image_prior_noise
            output["aux_regular_noise"] = aux_regular_noise

            alpha_grad = 4e-3 if step <= 60000 else 2e-3
            alpha_image = 0 if step <= 60000 else 2-1

            loss_total_noise = (alpha_grad * grad_prior_noise 
                              + alpha_image * image_prior_noise 

                              + aux_regular_noise)
            output["loss_total_noise"] = loss_total_noise

            loss_match = loss_match(loss_total_gt, loss_total_noise)
            output["loss_match"] = loss_match

            optimizer.zerp_grad()
            loss_match.backward()
            optimizer.step()
            scheduler.step()

            # --- log losses
            if step % 100 == 1:
                for (los_n, los_v) in output.items():
                    wandb.log({los_n: los_v}, step)

            # --- save images ---
            if step % 10 == 1:
                img_gt = img_gt[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                inp_noise = inp_noise[0].permute(1, 2, 0).detach().cpu().numpy() * 255
                imgs = np.concatenate((img_gt, inp_noise), 1)[:, :, ::-1]
                wandb.log({str(step) + "/img": wandb.Image(imgs)}, step)

            step += 1


if __name__ == "__main__":

    DATA_PATH = "./gradvit/data4GradVit"
    model_type = "vit-small" # mjp, vit-small
    model_path = ""

    # --- data ---
    dataset = GradVitDataset(
        root_path = DATA_PATH
    )
    data = DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        num_workers=8,
        drop_last=True,
        shuffle=False
    )

    # --- loss match ---
    loss_match = nn.L1Loss(reduction = 'mean')

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
        net = create_model("vit_small_patch16_224", pretrained=True)
        # net = create_model("vit_small_patch16_224", pretrained=True).cuda()


    runGradVit(model=net, 
               data=data, 
               loss_match=loss_match, 
               grad_prior=None, 
               image_prior=image_loss, 
               aux_regular=None)
