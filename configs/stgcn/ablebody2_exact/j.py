model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        tcn_dropout=0.5,
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=1,
        in_channels=256,
        loss_cls=dict(type='MSELoss', loss_weight=1.0)),
    test_cfg=dict(average_clips='score'))
dataset_type = 'PoseDataset'
ann_file = '/home/students/mmasabo1/summer26Research/ablebody2_exact.pkl'
pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='PadTo', length=300, mode='zero'),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=pipeline, split='test'))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=0.001, step=[2, 10])
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics=['mean_squared_error', 'mean_absolute_error'],
    save_best='mean_absolute_error',
    rule='less')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/stgcn/ablebody2_exact/joint'
