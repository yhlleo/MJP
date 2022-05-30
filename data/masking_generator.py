"""
Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
Copyright Zhun Zhong & Liang Zheng
Hacked together by / Copyright 2020 Ross Wightman
Modified by Hangbo Bao, for generating the masked position for visual image transformer
"""
# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# Originally inspired by impl at https://github.com/zhunzhong07/Random-Erasing, Apache 2.0
# Copyright Zhun Zhong & Liang Zheng
#
# Hacked together by / Copyright 2020 Ross Wightman
#
# Modified by Hangbo Bao, for generating the masked position for visual image transformer
# --------------------------------------------------------'
import random
import math
import copy
import numpy as np

import torch

class MaskingGenerator:
    def __init__(
            self, input_size, num_masking_patches, min_num_patches=8, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.init_num_masking_patches = num_masking_patches
        self.num_masking_patches = num_masking_patches

        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self):
        mask = np.zeros(shape=self.get_shape(), dtype=np.int32)
        mask_count = 0
        while mask_count < self.num_masking_patches:
            max_mask_patches = self.num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


    def _num_patches_decaying(self, total_epoch=300, cur_epoch=0, masking_ratio=-1):
        #max_ratio, min_ratio = 0.3, 0.1
        #cur_ratio = (max_ratio - min_ratio) * cur_epoch / total_epoch + min_ratio
        cur_ratio = masking_ratio if masking_ratio > 0 else 0.1
        self.num_masking_patches = int(self.num_patches * cur_ratio)


def patchmask2imagemask(mask, img_size=224, patch_size=16):
    num_patches_h = int(img_size // patch_size)
    num_patches_w = int(img_size // patch_size)

    img_mask = np.ones((num_patches_h, num_patches_w, patch_size, patch_size))
    mask = mask[:,:,np.newaxis, np.newaxis]

    img_mask = img_mask*mask
    img_mask = np.transpose(img_mask, (0,2,1,3)).reshape((img_size, img_size))
    return img_mask


class JigsawPuzzleMaskedRegion(object):
    def __init__(self, 
        img_size=224,
        patch_size=16,
        num_masking_patches=75,
        min_num_patches=8,
        mask_type="mjp" # "mjp" (ours), "pps" (MLP-Mixer, pixel shuffle per patch)
    ):
        input_size = int(img_size // patch_size)
        self.batch_size = 1
        self.masking_generator = MaskingGenerator(
            (input_size, input_size),
            num_masking_patches=num_masking_patches,
            min_num_patches=min_num_patches
        )

        self.im_to_patches = torch.nn.Unfold((patch_size,patch_size), stride=patch_size)
        self.patches_to_im = torch.nn.Fold(
            output_size=(img_size,img_size), 
            kernel_size=(patch_size,patch_size), 
            stride=patch_size
        )

        self.mask_type = mask_type

    def _update_masking_generator(self, total_epoch, epoch, masking_ratio=-1):
        self.masking_generator._num_patches_decaying(total_epoch, epoch, masking_ratio)

    def _get_masked_indexes(self):
        mask = self.masking_generator().flatten()
        nonzero = np.nonzero(mask)[0]
        nonzero_shuffle = copy.deepcopy(nonzero)
        np.random.shuffle(nonzero_shuffle)
        return mask, nonzero, nonzero_shuffle

    def _get_batch_masked_indexes(self):
        masks = self.masking_generator().flatten()[np.newaxis, :] # [1, C]
        cls_pad = np.zeros((1, 1), dtype=np.int32)
        masks = np.concatenate([cls_pad, masks], axis=1) # [1, C+1]
        return masks

    def __call__(self, x):
        """
        x: torch.Tensor, [N, C, H, W]
        """
        if self.mask_type == "mjp":
            N = x.size(0)
            to_patches = self.im_to_patches(x) # [N, C*patch_size*patch_size, seq_len]
            to_patches_copy = copy.deepcopy(to_patches)
            masks, nonzero, nonzero_shuffle = self._get_masked_indexes()
            to_patches[:,:,nonzero] = to_patches_copy[:,:,nonzero_shuffle]

            cls_pad = np.zeros((1, 1), dtype=np.int32)
            masks = np.concatenate([cls_pad, masks[np.newaxis, :]], axis=1) # [1, C+1]

            to_images = self.patches_to_im(to_patches)
            return to_images, masks
        else:
            N, C = x.size()[:2]
            to_patches = self.im_to_patches(x) # [N, C*patch_size*patch_size, seq_len]
            seq_len = to_patches.size(-1)
            to_patches = to_patches.view(N, C, -1, seq_len) # [N, C, patch_size*patch_size, seq_len]
            to_patches = to_patches[:,:,torch.randperm(to_patches.size(2))]
            to_patches = to_patches.view(N, -1, seq_len)
            to_patches = to_patches[:,:,torch.randperm(seq_len)]
            to_images = self.patches_to_im(to_patches)
            return to_images, None


if __name__ == '__main__':
    from PIL import Image
    from torchvision import transforms
    import torchvision.utils as vutils

    img_path = "/apdcephfs/share_916081/yliu/datasets/imagenet/train/n01558993/n01558993_1004.JPEG"

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])])
    img = transform(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
    jpwr = JigsawPuzzleMaskedRegion(mask_type='mjp')

    results = jpwr(img.repeat(4,1,1,1))[0]
    results = torch.cat([img, results],dim=0) * 0.5 + 0.5
    vutils.save_image(results.data, 'jpwr.png', nrow=results.size(0), padding=0)


    """
    import cv2

    input_size = 224
    patch_size = 16
    window_size = (input_size // patch_size, input_size // patch_size)
    masking_generator = MaskingGenerator(
        input_size=window_size,
        num_masking_patches=40,
        max_num_patches=None,
        min_num_patches=16
    )
    nonzero = np.nonzero(masking_generator().flatten() == 1)[0]
    np.random.shuffle(nonzero)
    print(nonzero)
    
    img_mask1 = patchmask2imagemask(np.array(masking_generator()).astype('float'))
    img_mask2 = patchmask2imagemask(np.array(masking_generator()).astype('float'))
    cv2.imshow("mask1", img_mask1)
    cv2.imshow("mask2", img_mask2)

    masking_generator._num_patches_decaying(300, 300)
    print(masking_generator.num_masking_patches)

    img_mask1 = patchmask2imagemask(np.array(masking_generator()).astype('float'))
    img_mask2 = patchmask2imagemask(np.array(masking_generator()).astype('float'))
    cv2.imshow("mask3", img_mask1)
    cv2.imshow("mask4", img_mask2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """