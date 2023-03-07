import pdb
import numpy as np
import wandb
import argparse
from tqdm import tqdm
from utils import reduce_tensor

import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.optim import lr_scheduler

from timm.data.transforms import _pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm import create_model

from models.vit_timm import vit_small_patch16_224
from gradvit_losses import grad_loss, image_loss, aux_patch_loss, aux_extra_loss, grad_loss_breaching
from data.masking_generator import JigsawPuzzleMaskedRegion


DATA_PATH = "/data/datasets/imagenet/val"
model_path = ""
USE_MJP = False
BATCH_SIZE = 1
model_type = "vit-small" # mjp, vit-small
DEBUG = False
STEP = 120000

def build_transform():
    IMG_SIZE = 224
    INTERPOLATION = 'bicubic'
    t = []
    size = int((256 / 224) * IMG_SIZE)
    t.append(
        transforms.Resize(size, interpolation=_pil_interp(INTERPOLATION)),
        # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(IMG_SIZE))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

dataset_val = datasets.ImageFolder(DATA_PATH, transform=build_transform())
data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=False
)

# load pretrained model
if model_type == "mjp":
    if DEBUG:
        net = vit_small_patch16_224(
            pretrained=False, 
            num_classes=1000,
            use_unk=True,
            use_idx_emb=False,
            use_dlocr=True,
            dlocr_type='nonlinear',  # nonlinear, linear, pca
            use_avg=False
        )
    else:
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
    if DEBUG: 
        net = create_model("vit_small_patch16_224", pretrained=True)
    else:
        net = create_model("vit_small_patch16_224", pretrained=True).cuda()
    # net = create_model("vit_small_patch16_224", pretrained=False,
    #     checkpoint_path="/path/to/pretrained_models/deit/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    # )

masking_ratio = 0.9
jigsaw_pullzer = JigsawPuzzleMaskedRegion(224, 16)
jigsaw_pullzer._update_masking_generator(300, 0, masking_ratio)

def validate(data_loader, model, jigsaw_pullzer, DEBUG, use_mjp=False, grad_loss_breaching=None, image_prior=None, aux_patch_loss=None, aux_extra_loss=None):
    
    wandb.init(project='MJP', entity="renbin", name="gradVit_attack")

    for idx, (images, target) in tqdm(enumerate(data_loader)):
        if DEBUG: 
            inp_noise = torch.randn(images.size())
            inp_noise.requires_grad=True
        else:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            inp_noise = torch.rand(images.size()).cuda()
            inp_noise.requires_grad=True

        # --- optimizer & lr_scheduler ---
        optimizer = torch.optim.Adam(
            [inp_noise],
            lr=0.1
        )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=STEP, 
            eta_min=0
        )

        model = model if DEBUG else model.cuda()

        # --- image processing with MJP ---
        unk_mask = None
        if use_mjp:
            images, unk_mask = jigsaw_pullzer(images)
            if DEBUG:
                unk_mask = torch.from_numpy(unk_mask).long()
            else:
                unk_mask = torch.from_numpy(unk_mask).long().cuda()

        # --- get input_gradient ---
        model.zero_grad()
        loss_fn = torch.nn.CrossEntropyLoss()
        ouputs_gt = model(images).sup
        loss_gt = loss_fn(ouputs_gt, target)
        inp_gradient = torch.autograd.grad(loss_gt, model.parameters())
        inp_gradient = [grad.detach() for grad in inp_gradient]
        # full_norm = torch.stack([g.norm() for g in inp_gradient]).mean()
 
        # --- GradVit attact ---
        step = 0
        while step <= STEP:
            print("STEP:", step)
            # --- 1.grad losses ---
            grad_prior_noise = (
                grad_loss_breaching(inp_gradient,
                    ouputs_gt,
                    inp_noise,
                    optimizer,
                    model,
                    jigsaw_pullzer,
                    DEBUG,
                    use_mjp
                )
                if grad_loss_breaching is not None
                else 0
            )

            # --- 2.image losses ---
            image_prior_noise = image_prior(inp_noise, DEBUG) if image_prior is not None else 0.

            # --- 3.aux_Reg losses ---
            aux_patch_noise = aux_patch_loss(inp_noise, DEBUG) if aux_patch_loss is not None else 0.
            aux_registration = 0.
            aux_extra_priors = aux_extra_loss(inp_noise) if aux_extra_loss is not None else 0.
            alpha1, alpha2, alpha3 = 1e-4, 1e-2, 1e-4

            aux_loss = (alpha1 * aux_patch_noise
                      + alpha2 * aux_registration
                      + alpha3 * aux_extra_priors
            ) 

            alpha_grad = 4e-4 if (step <= (STEP / 2)) else 2e-4
            alpha_image = 0 if (step <= (STEP / 2 )) else 2e-1

            # --- total losses ---
            loss_total_noise = 0.

            loss_total_noise = (alpha_grad * grad_prior_noise 
                              + alpha_image * image_prior_noise 
                              + aux_loss
            )

            output = {"grad_prior_noise": grad_prior_noise * alpha_grad}
            output["image_prior_noise"] = image_prior_noise * alpha_image
            output["aux_patch_noise"] = aux_patch_noise * alpha1
            output["aux_registration"] = aux_registration * alpha2
            output["aux_extra_priors"] = aux_extra_priors * alpha3
            output["aux_loss"] = aux_loss 

            output["loss_total_noise"] = loss_total_noise

            # --- optimize ---
            optimizer.zero_grad()
            loss_total_noise.backward()
            optimizer.step()
            scheduler.step()

            # --- log losses ---
            if step % 100 == 1:
                for (los_n, los_v) in output.items():
                    wandb.log({los_n: los_v}, step)
                    
            # --- save images ---
            if step % 100 == 1:    
                mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
                gt_image_vis = images[0].clone().detach()
                inp_noise_vis = inp_noise[0].clone().detach()
                for i in range(3):
                    gt_image_vis[i] = gt_image_vis[i] * std[i] + mean[i]
                    inp_noise_vis[i] = inp_noise_vis[i] * std[i] + mean[i]

                vis_gt = (gt_image_vis.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
                vis_inp_noise = (inp_noise_vis.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)

                wandb.log({str("gt") + "/img_gt": wandb.Image(vis_gt)}, step)
                wandb.log({str("noise") + "/img_noise": wandb.Image(vis_inp_noise)}, step)
    
            step += 1


if __name__ == "__main__":

    validate(
        data_loader_val, 
        net,
        jigsaw_pullzer,
        DEBUG,
        USE_MJP,
        grad_loss_breaching=grad_loss_breaching, 
        image_prior=image_loss, 
        aux_patch_loss=aux_patch_loss,
        aux_extra_loss=aux_extra_loss  
    )