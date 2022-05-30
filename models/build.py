# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin import SwinTransformer
from .resnet import ResNet50
from .vit_timm import (
    vit_tiny_patch16_224,
    vit_small_patch16_224,
    vit_base_patch16_224,
    deit_tiny_patch16_224,
    deit_small_patch16_224,
    deit_base_patch16_224
)


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_unk=config.TRAIN.USE_UNK_POS,
            use_idx_emb=config.TRAIN.USE_IDX_EMB,
            use_dlocr=config.TRAIN.USE_DLOCR,
            dlocr_type=config.TRAIN.DLOCR_TYPE
        )
    elif model_type == 'resnet50':
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES
        )
    elif model_type == "vit_s_16":
        model = vit_small_patch16_224(
            pretrained=False, 
            num_classes=config.MODEL.NUM_CLASSES,
            use_unk=config.TRAIN.USE_UNK_POS,
            use_idx_emb=config.TRAIN.USE_IDX_EMB,
            use_dlocr=config.TRAIN.USE_DLOCR,
            dlocr_type=config.TRAIN.DLOCR_TYPE
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
