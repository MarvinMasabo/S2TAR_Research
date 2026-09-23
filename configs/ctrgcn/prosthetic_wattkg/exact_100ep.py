# CTR-GCN joint regression on the prosthetic (TAMUSA / PRO_A&M) cohort, target = exact
# energy (Watts) / weight (kg). Same recipe as configs/ctrgcn/ablebody2_wattkg/exact_100ep.py
# (see that file for the full list of STGCN->CTRGCN adaptations); only the data source changed:
# prosthetic_wattkg/exact.pkl, built by build_prosthetic_pkl.py from final_demographics.pkl
# (785 clips, 3 participants: Knowles/Thorton/Richard; same schema as AbleBody2).
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='CTRGCN',
        graph_cfg=dict(layout='coco', mode='spatial'),
        num_person=1),   # CTRGCN (unlike STGCN) always sizes data_bn as num_person*C*V; defaults to
                         # 2 for NTU's multi-person clips, must be 1 to match our single-person data
    cls_head=dict(
        type='GCNHead',
        num_classes=1,
        in_channels=256,
        loss_cls=dict(type='MSELoss', loss_weight=1.0)),
    test_cfg=dict(average_clips='score'))   # required or GCNHead preds collapse to a constant

dataset_type = 'PoseDataset'
ann_file = '/home/students/mmasabo1/summer26Research/prosthetic_wattkg/exact.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint', 'label'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=10),
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
        dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy - cosine schedule spans the full 100 epochs
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 100
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,                              # validate every epoch -> full curve for the plot
    metrics=['mean_squared_error', 'mean_absolute_error'],
    save_best='mean_absolute_error',
    rule='less')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/ctrgcn/prosthetic_wattkg/exact_100ep'
