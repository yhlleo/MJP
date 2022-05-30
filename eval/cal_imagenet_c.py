import copy
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
from torch.autograd import Variable as V
import torchvision.datasets as dset

from models.vit_timm import vit_small_patch16_224

import argparse
parser = argparse.ArgumentParser(description='Evaluates robustness of various nets on ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', type=str)
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
args = parser.parse_args()
print(args)

PATH_TO_IMAGENET_VAL = "/path/to/imagenet/val"
PATH_TO_SAVE_INC = "/path/to/save/imagenet-c/"

torch.manual_seed(1)
np.random.seed(1)
if args.ngpu > 0:
    torch.cuda.manual_seed(1)
cudnn.benchmark = True  # fire on all cylinders

args.test_bs = 64
args.prefetch = 4

for p in net.parameters():
    p.volatile = True


# /////////////// Data Loader ///////////////

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

clean_loader = torch.utils.data.DataLoader(dset.ImageFolder(
    root=PATH_TO_IMAGENET_VAL,
    transform=trn.Compose([trn.Resize(256), trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])),
    batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

# load pretrained model
net = vit_small_patch16_224(
    pretrained=False, 
    num_classes=1000,
    use_unk=True,
    use_idx_emb=False,
    use_dlocr=True,
    dlocr_type='nonlinear',  # nonlinear, linear, pca
    use_avg=False
).cuda()
checkpoint = torch.load(args.model_path, map_location="cpu")
net.load_state_dict(checkpoint['model'], strict=True)
net.eval()

# /////////////// Further Setup ///////////////

def auc(errs):  # area under the distortion-error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area

correct = 0
for batch_idx, (data, target) in tqdm(enumerate(clean_loader)):
    data = V(data.cuda(), volatile=True)

    output = net(data).sup

    pred = output.data.max(1)[1]
    correct += pred.eq(target.cuda()).sum()

clean_error = 1 - float(correct) / len(clean_loader.dataset)
print('Clean dataset error (%): {:.2f}'.format(100 * clean_error))


def show_performance(distortion_name):
    errs = []

    for severity in range(1, 6):
        distorted_dataset = dset.ImageFolder(
            root=PATH_TO_SAVE_INC + distortion_name + '/' + str(severity),
            transform=trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.test_bs, shuffle=False, num_workers=args.prefetch, pin_memory=True)

        correct = 0
        for batch_idx, (data, target) in tqdm(enumerate(distorted_dataset_loader)):
            data = V(data.cuda(), volatile=True)

            output = net(data).sup

            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()

        errs.append(1 - 1.*correct / len(distorted_dataset))

    print('\n=Average', tuple(errs))
    return np.mean(errs)

# /////////////// Display Results ///////////////
import collections

print('\nUsing ImageNet data')

distortions = [
    'gaussian_noise', 
    'shot_noise', 
    'impulse_noise',
    'defocus_blur', 
    'glass_blur'
    #, 'motion_blur', 'zoom_blur',
    #'snow', 'frost', 'fog', 'brightness',
    #'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    #'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'
]

error_rates = []
for distortion_name in distortions:
    rate = show_performance(distortion_name)
    error_rates.append(rate)
    print('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))

print('mCE (unnormalized by AlexNet errors) (%): {:.2f}'.format(100 * np.mean(error_rates)))
