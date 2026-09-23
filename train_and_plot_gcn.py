"""Train a GCN model on the AbleBody2 "Exact Windowing" energy-expenditure target for
100 epochs, then plot the train-vs-validation MSE curve and report MAE/MSE/RMSE/r/R2 for
train/val/test at the best (lowest validation MAE) epoch.

Supports both models used in this project -- STGCN and CTR-GCN -- toggled by ONE line
(see "MODEL = " right below the imports). Everything else in the file is shared; the only
real differences between the two models are called out at MODEL_CFG below.

This is a self-contained, in-process equivalent of what actually produced this project's
results, e.g. for STGCN:
    bash tools/dist_train.sh configs/stgcn/ablebody2_wattkg/exact_100ep.py 1 \
        --validate --test-last --test-best --seed 0
`tools/dist_train.sh` launches pyskl's `tools/train.py` under `torch.distributed.launch`
(pyskl always wraps the model in MMDistributedDataParallel, even for one GPU), reading the
model/data/optimizer setup from a separate config file. Everything from both of those is
inlined below into one file: the config is a Python dict instead of a separate .py file, and
the one-GPU distributed process group that `torch.distributed.launch` would normally set up
is created directly with plain environment variables, so this runs as a single `python`
process with no shell wrapper.

============================================================================================
 TO SWITCH MODELS: change the one line below, nothing else needs to change.
============================================================================================
Run:
    python train_and_plot_gcn.py
"""
import glob
import os
import os.path as osp
import re
import time
import warnings
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import mmcv
from mmcv import Config
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, set_random_seed

from pyskl.apis import init_random_seed, train_model
from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import collect_env, get_root_logger

warnings.filterwarnings('ignore')

# ============================================================================
# THE ONE LINE TO CHANGE: 'STGCN' or 'CTRGCN'. Run once with each to compare.
# ============================================================================
MODEL = 'STGCN'

# What actually differs between the two models -- everything else in this file
# (data, optimizer, epochs, evaluation, plotting) is identical for both.
MODEL_CFG = {
    'STGCN': dict(
        backbone=dict(type='STGCN', graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
        # STGCN's data_bn defaults to 'VC' (value x channel) sizing, which is person-count
        # agnostic -- no num_person override needed here.
    ),
    'CTRGCN': dict(
        backbone=dict(type='CTRGCN', graph_cfg=dict(layout='coco', mode='spatial'), num_person=1),
        # Unlike STGCN, CTR-GCN always sizes its input batchnorm as num_person*channels*joints
        # (defaults to num_person=2, built for NTU's multi-person clips). Left at the default,
        # it crashes the moment real (single-person) data goes through -- num_person=1 is
        # required for our data, not optional/stylistic.
    ),
}
assert MODEL in MODEL_CFG, f'MODEL must be one of {list(MODEL_CFG)}'

ROOT = '/home/students/mmasabo1/summer26Research'
ANN_FILE = os.path.join(ROOT, 'ablebody2_wattkg/exact.pkl')
WORK_DIR = os.path.join(ROOT, f'work_dirs/{MODEL.lower()}/ablebody2_wattkg/exact_100ep_selfcontained')
SEED = 0
TOTAL_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------- config
def build_config():
    """Same model / data / optimizer setup as
    configs/{stgcn,ctrgcn}/ablebody2_wattkg/exact_100ep.py, inlined as a plain dict so this
    file has no separate config-file dependency. The backbone comes from MODEL_CFG above;
    data/optimizer/schedule are identical for both models (the same tamed lr=0.01 recipe
    that's already proven stable for STGCN was deliberately reused for CTR-GCN too, instead
    of CTR-GCN's own NTU-benchmark lr=0.1, which is tuned for a much larger classification
    dataset and is a known way to destabilize training on our much smaller regression set --
    see the PoseC3D lr=0.05 finding elsewhere in this project).
    """
    train_pipeline = [
        dict(type='PreNormalize2D'),
        dict(type='GenSkeFeat', dataset='coco', feats=['j']),
        dict(type='UniformSample', clip_len=100),
        dict(type='PoseDecode'),
        dict(type='FormatGCNInput', num_person=1),
        dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['keypoint', 'label']),
    ]
    val_pipeline = [
        dict(type='PreNormalize2D'),
        dict(type='GenSkeFeat', dataset='coco', feats=['j']),
        dict(type='UniformSample', clip_len=100, num_clips=1),
        dict(type='PoseDecode'),
        dict(type='FormatGCNInput', num_person=1),
        dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['keypoint']),
    ]
    test_pipeline = [
        dict(type='PreNormalize2D'),
        dict(type='GenSkeFeat', dataset='coco', feats=['j']),
        dict(type='UniformSample', clip_len=100, num_clips=10),
        dict(type='PoseDecode'),
        dict(type='FormatGCNInput', num_person=1),
        dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
        dict(type='ToTensor', keys=['keypoint']),
    ]
    cfg_dict = dict(
        model=dict(
            type='RecognizerGCN',
            backbone=MODEL_CFG[MODEL]['backbone'],
            cls_head=dict(
                type='GCNHead', num_classes=1, in_channels=256,
                loss_cls=dict(type='MSELoss', loss_weight=1.0)),
            test_cfg=dict(average_clips='score')),   # required, or GCNHead preds collapse to a constant
        dataset_type='PoseDataset',
        data=dict(
            videos_per_gpu=16,
            workers_per_gpu=2,
            test_dataloader=dict(videos_per_gpu=1),
            train=dict(
                type='RepeatDataset', times=5,
                dataset=dict(type='PoseDataset', ann_file=ANN_FILE, split='train', pipeline=train_pipeline)),
            val=dict(type='PoseDataset', ann_file=ANN_FILE, split='val', pipeline=val_pipeline),
            test=dict(type='PoseDataset', ann_file=ANN_FILE, split='test', pipeline=test_pipeline)),
        optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True),
        optimizer_config=dict(grad_clip=dict(max_norm=40, norm_type=2)),
        lr_config=dict(policy='CosineAnnealing', min_lr=0, by_epoch=False),
        total_epochs=TOTAL_EPOCHS,
        checkpoint_config=dict(interval=1),
        evaluation=dict(
            interval=1,                              # validate every epoch -> full curve for the plot
            metrics=['mean_squared_error', 'mean_absolute_error'],
            save_best='mean_absolute_error',
            rule='less'),
        log_config=dict(interval=20, hooks=[dict(type='TextLoggerHook')]),
        log_level='INFO',
        work_dir=WORK_DIR,
        dist_params=dict(backend='nccl' if torch.cuda.is_available() else 'gloo'),
    )
    return Config(cfg_dict)


# --------------------------------------------------------------------------- training
def run_training():
    """In-process equivalent of:
        bash tools/dist_train.sh configs/<model>/ablebody2_wattkg/exact_100ep.py 1
            --validate --test-last --test-best --seed 0
    """
    cfg = build_config()

    # pyskl's train_model() always wraps the model in MMDistributedDataParallel, so a
    # process group is required even for a single GPU -- this is what
    # `torch.distributed.launch --nproc_per_node=1` normally sets up before calling
    # tools/train.py. Doing it by hand here keeps everything in one process/file.
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', str(29500 + os.getpid() % 1000))
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['LOCAL_RANK'] = '0'
    init_dist('pytorch', **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    logger.info(f'Model: {MODEL}')
    logger.info('Environment info:\n' + '\n'.join(f'{k}: {v}' for k, v in collect_env().items()))
    logger.info(f'Config:\n{cfg.pretty_text}')

    seed = init_random_seed(SEED)
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=False)
    cfg.seed = seed
    meta = dict(seed=seed)

    model = build_model(cfg.model)
    datasets = [build_dataset(cfg.data.train)]
    cfg.workflow = cfg.get('workflow', [('train', 1)])
    cfg.checkpoint_config.meta = dict(config=cfg.pretty_text)

    dist.barrier()
    train_model(
        model, datasets, cfg,
        validate=True,
        test=dict(test_last=True, test_best=True),
        timestamp=timestamp, meta=meta)
    dist.barrier()
    return cfg


# --------------------------------------------------------------------------- log parsing
def epoch_train_mse(log_paths):
    """Mean batch loss_cls per epoch -> train MSE (MSELoss with loss_weight=1.0)."""
    per = {}
    for lp in sorted(log_paths):
        for line in open(lp):
            m = re.search(r'Epoch \[(\d+)\]\[.*loss_cls: ([0-9.]+)', line)
            if m:
                per.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return {e: float(np.mean(v)) for e, v in per.items()}


def epoch_val_metrics(log_paths):
    """pyskl logs regression val MSE in the 'mean_class_accuracy' field (reused from
    its classification-metric naming, since this codebase wasn't built for regression)."""
    mse, mae = {}, {}
    for lp in sorted(log_paths):
        for line in open(lp):
            m = re.search(r'Epoch\(val\) \[(\d+)\].*mean_class_accuracy: ([0-9.]+), '
                          r'mean_absolute_error: ([0-9.]+)', line)
            if m:
                e = int(m.group(1))
                mse[e] = float(m.group(2))
                mae[e] = float(m.group(3))
    return mse, mae


# --------------------------------------------------------------------------- evaluation
def build_eval_loader(cfg, split, spg=8):
    """Deterministic (num_clips=1) loader for any split, from the cfg.data.val template."""
    ds_cfg = deepcopy(cfg.data.val)
    ds_cfg['split'] = split
    ds_cfg['test_mode'] = True
    dataset = build_dataset(ds_cfg, dict(test_mode=True))
    return build_dataloader(dataset, videos_per_gpu=spg, workers_per_gpu=2, shuffle=False)


@torch.no_grad()
def run_inference(model, loader):
    preds, labels = [], []
    for batch in loader:
        y = batch['label'].numpy().reshape(-1)
        out = model(batch['keypoint'].to(DEVICE), return_loss=False)
        p = np.asarray(out).reshape(len(y), -1).mean(axis=1)
        preds.append(p); labels.append(y)
    return np.concatenate(preds), np.concatenate(labels)


def metrics(p, y):
    e = p - y
    ss_res, ss_tot = np.sum(e ** 2), np.sum((y - y.mean()) ** 2)
    mse = float(np.mean(e ** 2))
    return dict(MAE=float(np.mean(np.abs(e))), MSE=mse, RMSE=float(np.sqrt(mse)),
                r=float(np.corrcoef(p, y)[0, 1]), R2=float(1 - ss_res / ss_tot))


# --------------------------------------------------------------------------- report
def plot_and_report(cfg):
    logs = glob.glob(os.path.join(cfg.work_dir, '*.log'))
    tr = epoch_train_mse(logs)
    vmse, vmae = epoch_val_metrics(logs)
    epochs = sorted(set(tr) & set(vmse))
    best_ep = min(vmae, key=vmae.get)

    # curve plot: full range (left) + zoomed past epoch 1's huge starting loss (right)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    later_epochs = [e for e in epochs if e >= 3]
    zoom_from = min(later_epochs) if later_epochs else epochs[0]  # short runs (<3 epochs): no zoom needed
    for ax, (lo, title) in zip(axes, [(epochs[0], 'full range'), (zoom_from, f'zoomed, epoch {zoom_from}+')]):
        es = [e for e in epochs if e >= lo]
        ax.plot(es, [tr[e] for e in es], 'o-', ms=3, label='train MSE (mean batch loss)')
        ax.plot(es, [vmse[e] for e in es], 's-', ms=3, label='val MSE')
        if best_ep >= lo:
            ax.axvline(best_ep, color='r', ls='--', lw=1,
                       label=f'best epoch {best_ep} (val MAE {vmae[best_ep]:.4f})')
        ax.set_xlabel('epoch'); ax.set_ylabel('MSE, (W/kg)^2'); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle(f'{MODEL} / Exact Windowing — train vs val — FINAL ({len(epochs)}/{cfg.total_epochs} epochs)')
    plt.tight_layout()
    out_png = os.path.join(ROOT, f'{MODEL.lower()}_exact_100ep_curve_selfcontained.png')
    plt.savefig(out_png, dpi=120)
    print(f'wrote {out_png}')

    # train/val/test metrics at the best epoch -- needs a real forward pass, not just the log
    ckpt = os.path.join(cfg.work_dir, f'epoch_{best_ep}.pth')
    model = build_model(cfg.model).to(DEVICE).eval()
    load_checkpoint(model, ckpt, map_location='cpu')

    rows = []
    for split in ['train', 'val', 'test']:
        p, y = run_inference(model, build_eval_loader(cfg, split))
        rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **metrics(p, y)))
    report = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, f'{MODEL.lower()}_exact_100ep_report_selfcontained.csv')
    report.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}\n')
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(report.to_string(index=False))


if __name__ == '__main__':
    print(f'=== Training {MODEL} for {TOTAL_EPOCHS} epochs on Exact Windowing ===')
    trained_cfg = run_training()
    plot_and_report(trained_cfg)
