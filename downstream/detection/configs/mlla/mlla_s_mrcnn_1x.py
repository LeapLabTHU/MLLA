_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = './data/MLLA_S.pth'
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='MLLA',
        img_size=224, 
        patch_size=4, 
        in_chans=3,
        num_classes=80,
        embed_dim=64,
        depths=[3, 6, 21, 6],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4,
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        drop_path_rate=0.1,
        use_checkpoint=False,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(in_channels=[64, 128, 256, 512]))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
lr = 1e-4
bsz_per_gpu = 2
n_workers = 8
data = dict(
    train=dict(pipeline=train_pipeline),
    train_dataloader=dict(
        samples_per_gpu=bsz_per_gpu,
        workers_per_gpu=n_workers,
        pin_memory=True
    ))

optimizer = dict(_delete_=True, type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'an_bias': dict(decay_mult=0.),
                                                 'na_bias': dict(decay_mult=0.),
                                                 'ah_bias': dict(decay_mult=0.),
                                                 'aw_bias': dict(decay_mult=0.),
                                                 'ha_bias': dict(decay_mult=0.),
                                                 'wa_bias': dict(decay_mult=0.)}))
lr_config = dict(step=[8, 11])
runner = dict(max_epochs=12)