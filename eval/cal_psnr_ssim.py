import os
import sys
import glob
import argparse

from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.autograd import Variable

from basic import calc_psnr, calc_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--trg_dir', type=str)
parser.add_argument('--src_dir', type=str)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor()])

img_list = [f.split('/')[-1] for f in glob.glob(os.path.join(args.src_dir, "*.png"))]
source_list = [os.path.join(args.src_dir, f) for f in img_list]
target_list = [os.path.join(args.trg_dir, f) for f in img_list]

psnr_scores, ssim_scores = [], []
for f_src, f_trg in tqdm(zip(source_list, target_list)):
    if os.path.exists(f_src) and os.path.exists(f_trg):
        src_img = transform(Image.open(f_src).convert("RGB")).unsqueeze(0).cuda()
        trg_img = transform(Image.open(f_trg).convert("RGB")).unsqueeze(0).cuda()
        psnr = calc_psnr(src_img, trg_img).cpu().item()
        ssim = calc_ssim(src_img, trg_img).cpu().item()

        psnr_scores.append(psnr)
        ssim_scores.append(ssim)

print("PSRN: {:.5f} SSIM: {:.5f}".format(np.mean(psnr_scores), np.mean(ssim_scores)))