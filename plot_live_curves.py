"""Snapshot of in-progress (or finished) training curves, read from pyskl's logs.

    python plot_live_curves.py                      # merged prosthetic STGCN + CTR-GCN
    python plot_live_curves.py CONFIG [CONFIG ...]  # any runs

Writes live_training_curves.png; rerun (or loop it) to refresh -- VS Code reloads the image.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmcv import Config

import gcn_eval as ge

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ['configs/stgcn/prosthetic_wattkg/merged_100ep.py',
           'configs/ctrgcn/prosthetic_wattkg/merged_100ep.py']


def main():
    configs = sys.argv[1:] or DEFAULT
    fig, axes = plt.subplots(1, len(configs), figsize=(7 * len(configs), 5), squeeze=False)
    for ax, path in zip(axes[0], configs):
        cfg = Config.fromfile(os.path.join(ROOT, path))
        logs = glob.glob(os.path.join(ROOT, cfg.work_dir, '*.log'))
        tr = ge.epoch_train_mse(logs)
        vmse, vmae = ge.epoch_val_metrics(logs)
        name = f'{cfg.model.backbone.type} — {os.path.relpath(cfg.work_dir, "./work_dirs")}'
        if not vmse:
            ax.set_title(f'{name}\n(no validated epochs yet)')
            continue
        ep = sorted(vmse)
        ax.plot(sorted(tr), [tr[e] for e in sorted(tr)], '-', lw=2, label='train MSE')
        ax.plot(ep, [vmse[e] for e in ep], '-', lw=2, label='val MSE')
        best = min(vmae, key=vmae.get)
        ax.axvline(best, color='r', ls='--', lw=1, label=f'best so far: epoch {best} (val MAE {vmae[best]:.3f})')
        ax.set_yscale('log')
        ax.set_xlim(0, cfg.total_epochs)
        ax.set_xlabel('epoch')
        ax.set_ylabel('MSE, (W/kg)²  — log scale')
        ax.set_title(f'{name}\n{len(ep)}/{cfg.total_epochs} epochs')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    out = os.path.join(ROOT, 'live_training_curves.png')
    plt.savefig(out, dpi=110)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
