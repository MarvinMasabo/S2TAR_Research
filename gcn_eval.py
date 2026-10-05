"""Shared evaluation utilities for the regular pipeline's report step (see PIPELINE.md, step 4).

Used by report_stgcn_ablebody.py, report_ctrgcn_ablebody.py, and report_prosthetic.py.
Every one of them loads the *actual* config file a training run used
(mmcv.Config.fromfile(...)) and passes it to these functions -- there is no separate,
hand-maintained copy of the model/data/optimizer setup here, so a report can never drift
out of sync with what was really trained.
"""
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch

from pyskl.datasets import build_dataloader, build_dataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --------------------------------------------------------------------------- log parsing
def epoch_train_mse(log_paths):
    """Mean batch loss_cls per epoch -> train MSE (MSELoss with loss_weight=1.0).
    Read straight from the training log -- no extra inference needed for this part."""
    import re
    per = {}
    for lp in sorted(log_paths):
        for line in open(lp):
            m = re.search(r'Epoch \[(\d+)\]\[.*loss_cls: ([0-9.]+)', line)
            if m:
                per.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return {e: float(np.mean(v)) for e, v in per.items()}


def epoch_val_metrics(log_paths):
    """pyskl logs regression val MSE in the 'mean_class_accuracy' field (reused from its
    classification-metric naming, since this codebase wasn't built for regression)."""
    import re
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
                MRE=float(np.mean(np.abs(e) / np.abs(y))),
                r=float(np.corrcoef(p, y)[0, 1]), R2=float(1 - ss_res / ss_tot))


# --------------------------------------------------------------------------- plotting
def plot_train_vs_val(tr, vmse, vmae, best_ep, total_epochs, title, out_png):
    """The full-range + zoomed two-panel train-vs-val curve used for every run in this project."""
    epochs = sorted(set(tr) & set(vmse))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    later = [e for e in epochs if e >= 3]
    zoom_from = min(later) if later else epochs[0]   # short runs (<3 epochs): no zoom needed
    for ax, (lo, panel_title) in zip(axes, [(epochs[0], 'full range'), (zoom_from, f'zoomed, epoch {zoom_from}+')]):
        es = [e for e in epochs if e >= lo]
        ax.plot(es, [tr[e] for e in es], 'o-', ms=3, label='train MSE (mean batch loss)')
        ax.plot(es, [vmse[e] for e in es], 's-', ms=3, label='val MSE')
        if best_ep >= lo:
            ax.axvline(best_ep, color='r', ls='--', lw=1,
                       label=f'best epoch {best_ep} (val MAE {vmae[best_ep]:.4f})')
        ax.set_xlabel('epoch'); ax.set_ylabel('MSE, (W/kg)^2'); ax.set_title(panel_title); ax.legend(fontsize=8)
    fig.suptitle(f'{title} — train vs val — FINAL ({len(epochs)}/{total_epochs} epochs)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'wrote {out_png}')
