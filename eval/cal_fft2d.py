import os
import sys
import glob
import argparse

from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from scipy.spatial.distance import cosine

parser = argparse.ArgumentParser()
parser.add_argument('--trg_dir', type=str)
parser.add_argument('--src_dir', type=str)
args = parser.parse_args()

transform = transforms.Compose([
    transforms.ToTensor()])

img_list = [f.split('/')[-1] for f in glob.glob(os.path.join(args.src_dir, "*.png"))]
source_list = [os.path.join(args.src_dir,  f) for f in img_list]
target_list = [os.path.join(args.trg_dir,  f) for f in img_list]

fft2d_scores = []
for f_src, f_trg in tqdm(zip(source_list, target_list)):
    if os.path.exists(f_src) and os.path.exists(f_trg):
        src_img = transform(Image.open(f_src).convert("RGB")).unsqueeze(0)
        trg_img = transform(Image.open(f_trg).convert("RGB")).unsqueeze(0)

        src_img = np.fft.fft2(src_img.data.cpu().numpy())
        trg_img = np.fft.fft2(trg_img.data.cpu().numpy())

        fft2d = cosine(src_img.flatten(), trg_img.flatten())
        fft2d = np.sqrt(fft2d.real**2 + fft2d.imag**2)
        fft2d_scores.append(fft2d)

print("FFT2D: {:.5f}".format(np.mean(fft2d_scores)))