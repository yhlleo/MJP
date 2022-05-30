
import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from timm.data.transforms import _pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.vit_timm import vit_small_patch16_224
from models.mlp_mixer import mixer_s16
from models.pvt import pvt_small
from models.resnet import ResNet50
from models.swin import SwinTransformer
from models.t2t import T2t_vit_14 

from data.masking_generator import JigsawPuzzleMaskedRegion

model_type = "vit_s_small"
data_dir = "/path/to/imagenet/val"
model_path = "/path/to/model"
checkpoint = torch.load(model_path, map_location='cpu')

def vit_features(model, x, unk_mask=None):
    cls_token = model.forward_features(x, alpha, unk_mask)[0]
    logits = model.head(cls_token)
    return cls_token, logits

def swin_features(model, x, unk_mask=None):
    feat = model.norm(model.forward_features(x, alpha)[-1])  # [B, L, C]
    feat = model.avgpool(feat.transpose(1, 2))  # [B, C, 1]
    cls_token = torch.flatten(feat, 1)
    logits = model.head(cls_token)
    return cls_token, logits

model = vit_small_patch16_224(
    pretrained=False, 
    num_classes=1000,
    use_unk=True,
    use_idx_emb=False,
    use_dlocr=True,
    dlocr_type='nonlinear'  # nonlinear, linear, pca
).cuda()
model.eval()
model.load_state_dict(checkpoint['model'], strict=True)

feat_func = vit_features

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=_pil_interp('bicubic')),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

img_list = glob.glob(os.path.join(data_dir, "*/*.JPEG"))
im_to_patches = torch.nn.Unfold((16,16), stride=16)
patches_to_im = torch.nn.Fold(output_size=(224,224), kernel_size=(16,16), stride=16)

def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1).type(torch.FloatTensor).mean().item()
    return agree

masking_ratio = 0.5
jigsaw_pullzer = JigsawPuzzleMaskedRegion(224, 16)
jigsaw_pullzer._update_masking_generator(300, 0, masking_ratio)

scores, agrees = [], []
for fim in tqdm(img_list):
    img = Variable(transform(Image.open(fim).convert("RGB")).unsqueeze(0)).cuda()
    
    img_jigsaw, unk_mask = jigsaw_pullzer(img)
    unk_mask = torch.from_numpy(unk_mask).long().cuda()
    
    cls_token1, logits1 = feat_func(model, img, None)
    cls_token2, logits2 = feat_func(model, img_jigsaw, unk_mask)
    
    scores.append((cls_token1 - cls_token2).norm().item())
    agrees.append(agreement(logits1, logits2))
print(np.mean(scores), np.mean(agrees))








