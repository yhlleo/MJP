import cv2
import pdb
import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets

from timm.models.layers import PatchEmbed
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm import create_model

from data.masking_generator import JigsawPuzzleMaskedRegion

masking_ratio = 0.9
jigsaw_pullzer = JigsawPuzzleMaskedRegion(224, 16)
jigsaw_pullzer._update_masking_generator(300, 0, masking_ratio)


def grad_loss(images, target, inp_noise, model, jigsaw_pullzer, DEBUG, use_mjp=False):
    # --- process input with mjp ---
    unk_mask = None
    if use_mjp:
        images, unk_mask = jigsaw_pullzer(images)
        if DEBUG:
            unk_mask = torch.from_numpy(unk_mask).long()
        else:
            unk_mask = torch.from_numpy(unk_mask).long().cuda()

    ### --- For Gt Inputs --- ###
    # --- 1. get shared gradients ---
    grad_out_gt = []
    def register_backward_hook(module, gin, gout):
        grad_out_gt.append(gout)

    hooks_gt = []
    for i, (name, module) in enumerate(model.named_modules()):
        # if "mlp.drop" in name:
        handle = module.register_backward_hook(register_backward_hook)
        hooks_gt.append(handle)

    # --- 2. compute output ---
    output = model(images, unk_mask=unk_mask)

    # --- 3. crossentropy loss
    criterion_shared = nn.CrossEntropyLoss()
    loss_shard = criterion_shared(output.sup, target)

    # --- 4. backward loss ---
    model.zero_grad()
    loss_shard.backward()

    # --- 5. save ---
    grad_gt_save = [elem for elem in grad_out_gt]
    grad_out_gt.clear()
    
    ### --- For Noise Inputs --- ###
    # --- 1. compute output for noise input ---
    output_noise = model(inp_noise, unk_mask=unk_mask) 

    # --- 2. crossentropy loss ---
    criterion_synthesized = nn.CrossEntropyLoss()
    loss_noise = criterion_synthesized(output_noise.sup, output.sup.detach())

    # --- 4. backward loss ---
    model.zero_grad()
    loss_noise.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1)

    grad_noise_save = grad_out_gt
    # grad_out_gt.clear()

    ### --- For L2 Norm --- ###
    loss = torch.nn.MSELoss(reduction="mean") 
    loss_grad_out = 0.

    assert(len(grad_noise_save) == len(grad_gt_save))

    for i in range(0, len(grad_noise_save)):
        loss_grad_out += loss(grad_noise_save[i][0], grad_gt_save[i][0])

    grad_out_gt.clear()

    for i in range(len(hooks_gt)):
        hooks_gt[i].remove()

    return loss_grad_out


def grad_loss_breaching(inp_gradient, labels, inp_noise, optimizer, model, jigsaw_pullzer, DEBUG, use_mjp=False):

    optimizer.zero_grad()
    model.zero_grad()
    loss_fn = torch.nn.CrossEntropyLoss()
    loss_noise = loss_fn(model(inp_noise).sup, labels.detach())

    gradients = torch.autograd.grad(loss_noise, model.parameters(), create_graph=True)
    cost_fn = nn.MSELoss(reduction="mean")
    costs = 0.
    for i in range(len(gradients)):
        costs += cost_fn(gradients[i], inp_gradient[i]) 

    return costs
        
        
def image_loss(inp_noise, DEBUG):
    loss = torch.nn.MSELoss(reduction="mean")

    if DEBUG:
        model = models.resnet50(pretrained=False)
    else:
        model = models.resnet50(pretrained=False).cuda()

    checkpoint = torch.load(
                    "/data/bren/projects/privacy_FL/MJP/gradvit/moco_v2_200ep_pretrain.pth.tar" , 
                    map_location="cpu"
                )
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
        if 'bn' in name:
            module.register_forward_hook(bn_inp_hook)

    # --- inp_noise ---
    oup_noise = model(inp_noise)

    # --- get x_{\hat} mean and var --
    bn_mean_gt_list = []
    bn_var_noise_list = []
    for i in range(0, len(bn_inp)):
        mean = torch.mean(bn_inp[i][0], dim=[0, 2, 3])
        var = torch.var(bn_inp[i][0], dim=[0, 2, 3])
        bn_mean_gt_list.append(mean)
        bn_var_noise_list.append(var)

    # --- loss ---
    mean_diff = 0.
    var_diff = 0.
    for i in range(0, len(bn_run_mean_value_gt_list)):
        mean_diff += loss(bn_run_mean_value_gt_list[i], bn_mean_gt_list[i])
        var_diff += loss(bn_run_var_value_gt_list[i], bn_var_noise_list[i])

    image_loss = mean_diff + var_diff
    return image_loss


def aux_patch_loss(inp_noise, DEBUG):
    loss = torch.nn.MSELoss(reduction="mean")

    if DEBUG:
        patch_embedder = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
    else:
        patch_embedder = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768).cuda()

    inp_noise_patches = patch_embedder(inp_noise)
    B, _, _  = inp_noise_patches.shape
    inp_noise_patches = inp_noise_patches.reshape((B, 14, 14, 3, 16, 16))
    b, num_h, num_w, c, h_patch, w_patch = inp_noise_patches.shape

    loss_patch = 0.
    for i in range(num_h-1):
        loss_vertical = loss(inp_noise_patches[:, i+1, :, :, 0, :], inp_noise_patches[:, i, :, :, 15, :])
        loss_horizontal = loss(inp_noise_patches[:, :, i+1, :, :, 0], inp_noise_patches[:, :, i, :, :, 15])
        loss_patch += (loss_vertical + loss_horizontal)

    return loss_patch


def aux_extra_loss(inp_noise):
    tv_loss = TVLoss()
    out = tv_loss(inp_noise) + torch.norm(inp_noise, p=2)
    return out


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

if __name__ == "__main__":
    inp_gt = cv2.imread('/data/bren/projects/privacy_FL/MJP/gradvit/data4GradVit/user1.png')
    inp_gt = torch.from_numpy(inp_gt).permute(2, 0, 1)[None] / 255.
    
    inp_noise = torch.randn(inp_gt.size(), requires_grad=True)

    aux_extra_loss(inp_gt, inp_noise)
    net = create_model("vit_small_patch16_224", pretrained=True)
    # net = create_model("vit_small_patch16_224", pretrained=True).cuda()
