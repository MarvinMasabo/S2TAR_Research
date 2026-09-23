# CTR-GCN joint regression on AbleBody2, target = exact energy (Watts) / weight (kg).
# Base template: configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/j.py (closest available 2D-coco+HRNet
# CTRGCN config in pyskl; there's no project-specific ctrgcn_stmarys_hrnet template to clone).
# Adaptations for regression on our data:
#   - cls_head: num_classes 60->1, added loss_cls=MSELoss, added test_cfg=average_clips='score'
#     (required or GCNHead predictions collapse to a constant, same fix as STGCN/PoseC3D)
#   - FormatGCNInput num_person 2->1 (our clips are single-person, not NTU multi-person)
#   - backbone num_person 2->1: unlike STGCN, CTRGCN always sizes data_bn as num_person*C*V
#     (no VC-only mode) — left at the default 2 it crashes with a batchnorm channel mismatch
#     the moment a real (single-person) batch goes through
#   - ann_file/split -> our W/kg exact.pkl with 'train'/'val'/'test' (not NTU's xsub_train/xsub_val)
#   - train pipeline gains ToTensor(['keypoint','label']) so the label reaches MSELoss as a tensor
#   - optimizer lr 0.1->0.01 and weight_decay 0.0005->0.0001, matching the already-stable STGCN
#     wattkg recipe: lr=0.1 is the NTU benchmark's own tuned value for a much larger classification
#     dataset, and PoseC3D already showed that carrying over a "too hot" borrowed lr onto our much
#     smaller regression set causes the validation curve to oscillate instead of converge.
#   - grad_clip enabled (STGCN's setting) instead of the NTU template's grad_clip=None, as an extra
#     stability guard for the same reason.
# Long-run from the start (100 epochs, not 24): this is the "train longer, plot train vs val" task,
# same treatment as configs/stgcn/ablebody2_wattkg/exact_100ep.py, applied to CTRGCN.
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
ann_file = '/home/students/mmasabo1/summer26Research/ablebody2_wattkg/exact.pkl'
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
work_dir = './work_dirs/ctrgcn/ablebody2_wattkg/exact_100ep'
