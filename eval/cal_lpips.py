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

from lpips import LPIPS

parser = argparse.ArgumentParser()
parser.add_argument('--trg_dir', type=str)
parser.add_argument('--src_dir', type=str)
args = parser.parse_args()

device = torch.device('cuda')
lpips = LPIPS(device=device).eval().cuda()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])])

img_list = [f.split('/')[-1] for f in glob.glob(os.path.join(args.src_dir, "*.png"))]
source_list = [os.path.join(args.src_dir, f) for f in img_list]
target_list = [os.path.join(args.trg_dir, f) for f in img_list]

lpips_scores = []
for f_src, f_trg in tqdm(zip(source_list, target_list)):
    if os.path.exists(f_src) and os.path.exists(f_trg):
        src_img = Variable(transform(Image.open(f_src).convert("RGB")).unsqueeze(0).to(device))
        trg_img = Variable(transform(Image.open(f_trg).convert("RGB")).unsqueeze(0).to(device))
        
        tmp = lpips(src_img, trg_img).cpu().item()
        lpips_scores.append(tmp)

print("Overall Average:", np.mean(lpips_scores))
