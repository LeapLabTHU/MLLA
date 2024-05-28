# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------

from .mlla import MLLA


def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'mlla':
        model = MLLA(img_size=config.DATA.IMG_SIZE,
                     patch_size=config.MODEL.MLLA.PATCH_SIZE,
                     in_chans=config.MODEL.MLLA.IN_CHANS,
                     num_classes=config.MODEL.NUM_CLASSES,
                     embed_dim=config.MODEL.MLLA.EMBED_DIM,
                     depths=config.MODEL.MLLA.DEPTHS,
                     num_heads=config.MODEL.MLLA.NUM_HEADS,
                     mlp_ratio=config.MODEL.MLLA.MLP_RATIO,
                     qkv_bias=config.MODEL.MLLA.QKV_BIAS,
                     drop_rate=config.MODEL.DROP_RATE,
                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                     ape=config.MODEL.MLLA.APE,
                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
