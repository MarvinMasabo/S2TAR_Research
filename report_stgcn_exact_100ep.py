"""Report for the STGCN / Exact Windowing 100-epoch run:
  - train-vs-validation MSE curve (from the training log)
  - metrics (MSE/MAE/RMSE/r/R2) on train, val, and test for the best-val-MAE epoch

Safe to run mid-training: it just uses whatever epochs/checkpoints exist so far.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.models import build_model
from pyskl.datasets import build_dataset, build_dataloader

warnings.filterwarnings('ignore')

ROOT = '/home/students/mmasabo1/summer26Research'
WORK_DIR = os.path.join(ROOT, 'work_dirs/stgcn/ablebody2_wattkg/exact_100ep')
CONFIG = os.path.join(ROOT, 'configs/stgcn/ablebody2_wattkg/exact_100ep.py')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def epoch_train_mse(log_paths):
    per = {}
    for lp in sorted(log_paths):
        for line in open(lp):
            m = re.search(r'Epoch \[(\d+)\]\[.*loss_cls: ([0-9.]+)', line)
            if m:
                per.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return {e: float(np.mean(v)) for e, v in per.items()}


def epoch_val_metrics(log_paths):
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


def build_eval_loader(cfg, split, spg=8):
    ds_cfg = deepcopy(cfg.data.val)
    ds_cfg['type'] = ds_cfg.get('type', 'PoseDataset')
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
    ss_res = np.sum(e ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    mse = float(np.mean(e ** 2))
    return dict(MAE=float(np.mean(np.abs(e))), MSE=mse, RMSE=float(np.sqrt(mse)),
                r=float(np.corrcoef(p, y)[0, 1]), R2=float(1 - ss_res / ss_tot))


def main():
    logs = glob.glob(os.path.join(WORK_DIR, '*.log'))
    tr = epoch_train_mse(logs)
    vmse, vmae = epoch_val_metrics(logs)
    epochs = sorted(set(tr) & set(vmse))
    if not epochs:
        print('No completed epochs yet.')
        return
    last = epochs[-1]
    best_ep = min(vmae, key=vmae.get)
    total_epochs = Config.fromfile(CONFIG).total_epochs
    partial = last < total_epochs
    tag = f'PARTIAL ({last}/{total_epochs} epochs so far)' if partial else f'FINAL ({total_epochs}/{total_epochs} epochs)'

    # --- curve plot: full range (left) + zoomed past epoch 1's huge starting loss (right) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    zoom_from = min(e for e in epochs if e >= 3)
    for ax, (lo, title) in zip(axes, [(epochs[0], 'full range'), (zoom_from, f'zoomed, epoch {zoom_from}+')]):
        es = [e for e in epochs if e >= lo]
        ax.plot(es, [tr[e] for e in es], 'o-', ms=3, label='train MSE (mean batch loss)')
        ax.plot(es, [vmse[e] for e in es], 's-', ms=3, label='val MSE')
        if best_ep >= lo:
            ax.axvline(best_ep, color='r', ls='--', lw=1, label=f'best epoch {best_ep} (val MAE {vmae[best_ep]:.4f})')
        ax.set_xlabel('epoch'); ax.set_ylabel('MSE, (W/kg)^2'); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle(f'STGCN / Exact Windowing — train vs val — {tag}')
    plt.tight_layout()
    out_png = os.path.join(ROOT, 'stgcn_exact_100ep_curve.png')
    plt.savefig(out_png, dpi=120)
    print(f'wrote {out_png}  ({tag})')

    # --- best-epoch train/val/test report ---
    ckpt = os.path.join(WORK_DIR, f'epoch_{best_ep}.pth')
    if not os.path.exists(ckpt):
        ckpt = os.path.join(WORK_DIR, f'best_mean_absolute_error_epoch_{best_ep}.pth')
    if not os.path.exists(ckpt):
        print(f'No checkpoint on disk yet for best epoch {best_ep} — plot only.')
        return

    cfg = Config.fromfile(CONFIG)
    model = build_model(cfg.model).to(DEVICE).eval()
    load_checkpoint(model, ckpt, map_location='cpu')
    model.eval()

    rows = []
    for split in ['train', 'val', 'test']:
        p, y = run_inference(model, build_eval_loader(cfg, split))
        m = metrics(p, y)
        rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **m))
    report = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, 'stgcn_exact_100ep_report.csv')
    report.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}  ({tag})\n')
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(report.to_string(index=False))


if __name__ == '__main__':
    main()
