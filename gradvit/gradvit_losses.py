import cv2
import pdb
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

# class GradLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, inp, model):
        
#         out = inp
#         return out 


def image_loss(inp_noise):
    loss = torch.nn.MSELoss(reduction="sum")

    model = models.resnet50(pretrained=False)
        
    checkpoint = torch.load("/data/bren/projects/privacy_FL/MJP/gradvit/moco_v2_200ep_pretrain.pth.tar" , map_location="cpu")
    state_dict = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # --- get GT running_mean and running_var ---
    bn_run_mean_name_gt_list = []
    bn_run_mean_value_gt_list = []
    bn_run_var_name_gt_list = []
    bn_run_var_value_gt_list = []

    for name in model.state_dict():
        if ('.bn' and 'running_mean' in name) and ('sample' not in name):
            bn_run_mean_name_gt_list.append(name)
            bn_run_mean_value_gt_list.append(model.state_dict()[name])
        if ('.bn' and 'running_var' in name) and ('sample' not in name):
            bn_run_var_name_gt_list.append(name)
            bn_run_var_value_gt_list.append(model.state_dict()[name])

    bn_inp = []
    def bn_inp_hook(module, input, output):
        bn_inp.append(input)

    for (name, module) in model.named_modules():
        print(name)
        if 'bn' in name:
            module.register_forward_hook(bn_inp_hook)

    # --- inp_noise ---
    oup_noise = model(inp_noise)

    # --- get x_{\hat} mean and var ---
    #TODO Let's assume we already got the input list of of each BN layer
    # Questions: How to copy/convert tuple to tensor???!!!
    bn_mean_gt_list = []
    bn_var_noise_list = []
    for i in range(0, len(bn_inp)):
        mean = torch.mean(bn_inp[i], dim=0)
        var = torch.var(bn_inp[i], dim=0)
        bn_mean_gt_list.append(mean)
        bn_var_noise_list.append(var)

    # --- loss ---
    mean_diff = 0.
    var_diff = 0.

    for i in range(0, len(bn_run_mean_value_gt_list)):
        mean_diff += torch.sqrt((loss(bn_run_mean_value_gt_list[i], bn_mean_gt_list[i]) * len(bn_run_mean_value_gt_list)))
        var_diff += torch.sqrt((loss(bn_run_var_value_gt_list[i], bn_var_noise_list[i]) * len(bn_run_mean_value_gt_list)))

    image_loss = mean_diff + var_diff

    return image_loss


def aux_losses(inp_gt, inp_noise):
    """
    aux_losses = \alpha1(1e-4) * patch prior 
            + \alpha2(1e-2) * registration 
            + \alpha3(1e-4) * extra priors
    """
    # --- patch prior ---


    # --- registration ---


    # --- extra priors ---


# class AuxLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, inp, model):
        
#         out = inp
#         return out 

if __name__ == "__main__":
    inp_gt = cv2.imread('/data/bren/projects/privacy_FL/MJP/gradvit/data4GradVit/user1.png')
    inp_gt = torch.from_numpy(inp_gt).permute(2, 0, 1)[None] / 255.
    
    inp_noise = torch.randn(inp_gt.size(), requires_grad=True)

    # model = models.resnet50(pretrained=True)

    # state_dict = model.state_dict()

    # for name in state_dict:
    #     print(name)

    image_loss(inp_noise)

