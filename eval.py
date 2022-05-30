
import numpy as np
import argparse
import torch
from torchvision import transforms, datasets
from timm.data.transforms import _pil_interp
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from timm.utils import accuracy, AverageMeter
from tqdm import tqdm

from utils import reduce_tensor
from models.vit_timm import vit_small_patch16_224

from timm import create_model

from data.masking_generator import JigsawPuzzleMaskedRegion

DATA_PATH = ""
model_path = ""
USE_MJP = True
BATCH_SIZE = 128
model_type = "mjp" # mjp, vit-small

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
    net = create_model("vit_small_patch16_224", pretrained=False,
        checkpoint_path="/path/to/pretrained_models/deit/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz"
    )

net.eval()

masking_ratio = 0.9
jigsaw_pullzer = JigsawPuzzleMaskedRegion(224, 16)
jigsaw_pullzer._update_masking_generator(300, 0, masking_ratio)

@torch.no_grad()
def validate(data_loader, model, jigsaw_pullzer, use_mjp=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    for idx, (images, target) in tqdm(enumerate(data_loader)):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        unk_mask = None
        if use_mjp:
            images, unk_mask = jigsaw_pullzer(images)
            unk_mask = torch.from_numpy(unk_mask).long().cuda()

        # compute output
        output = model(images, unk_mask=unk_mask)

        # measure accuracy
        acc1, acc5 = accuracy(output.sup, target, topk=(1, 5))

        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

    print(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

validate(
    data_loader_val, 
    net,
    jigsaw_pullzer,
    USE_MJP
)