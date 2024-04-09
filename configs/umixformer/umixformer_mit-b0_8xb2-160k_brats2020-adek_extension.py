_base_ = [
    "../_base_/models/segformer_mit-b0.py",
    "../_base_/datasets/brats2020.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_brain_tumor.py",
]
checkpoint = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth"  # noqa
randomness = dict(seed=0)  # seed setup
find_unused_parameters = True  # find it in mmcv
crop_size = (128, 128)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=dict(type="Pretrained", checkpoint=checkpoint)),
    decode_head=dict(
        type="APFormerHead2",  #'APFormerHeadSingle' #FeedFormerHeadUNet, FeedFormerHeadUNetPlus, FeedFormerHead32, FeedFormerHead32_new'
        feature_strides=[4, 8, 16, 32],
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        decoder_params=dict(
            embed_dim=128, num_heads=[8, 5, 2, 1], pool_ratio=[1, 2, 4, 8]
        ),
        num_classes=2,
        out_channels=2,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.5, 0.5],
        ),
    ),
)

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            "pos_block": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "head": dict(lr_mult=10.0),
        }
    ),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=5000),
    dict(
        type="PolyLR",  # check CosineAnnealingLR: https://github.com/open-mmlab/mmengine/blob/04b0ffee76c41d10c5fd1f737cdc5306af365754/mmengine/optim/scheduler/lr_scheduler.py#L48
        eta_min=0.0,
        power=1.0,
        begin=5000,
        end=10000,
        by_epoch=False,
    ),
]

# val_evaluator = dict(type='SingleIoUMetric', iou_metrics=['mIoU']) #, output_dir='paper/mIoU')
# test_evaluator = val_evaluator

train_dataloader = dict(batch_size=16, num_workers=1)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = val_dataloader
