"""Train-vs-validation curve + best-epoch train/val/test report for the prosthetic-cohort
runs (STGCN and CTR-GCN, Exact Windowing). Same plotting code as train_and_plot_gcn.py's
plot_and_report() -- this script just evaluates an already-completed training run instead of
training one, and writes prosthetic-specific output filenames.
"""
import glob
import os

import pandas as pd
import torch
from mmcv.runner import load_checkpoint

import train_and_plot_gcn as M   # reuses epoch_train_mse, epoch_val_metrics, build_eval_loader,
                                  # run_inference, metrics, build_config, build_model, plt

ROOT = '/home/students/mmasabo1/summer26Research'
ANN_FILE = os.path.join(ROOT, 'prosthetic_wattkg/exact.pkl')


def report(model_name, work_dir):
    M.MODEL = model_name
    M.ANN_FILE = ANN_FILE
    M.WORK_DIR = work_dir
    cfg = M.build_config()

    logs = glob.glob(os.path.join(cfg.work_dir, '*.log'))
    tr = M.epoch_train_mse(logs)
    vmse, vmae = M.epoch_val_metrics(logs)
    epochs = sorted(set(tr) & set(vmse))
    best_ep = min(vmae, key=vmae.get)

    # ---- THE PLOT: same code as train_and_plot_gcn.py's plot_and_report() ----
    fig, axes = M.plt.subplots(1, 2, figsize=(14, 5.5))
    later = [e for e in epochs if e >= 3]
    zoom_from = min(later) if later else epochs[0]
    for ax, (lo, title) in zip(axes, [(epochs[0], 'full range'), (zoom_from, f'zoomed, epoch {zoom_from}+')]):
        es = [e for e in epochs if e >= lo]
        ax.plot(es, [tr[e] for e in es], 'o-', ms=3, label='train MSE (mean batch loss)')
        ax.plot(es, [vmse[e] for e in es], 's-', ms=3, label='val MSE')
        if best_ep >= lo:
            ax.axvline(best_ep, color='r', ls='--', lw=1,
                       label=f'best epoch {best_ep} (val MAE {vmae[best_ep]:.4f})')
        ax.set_xlabel('epoch'); ax.set_ylabel('MSE, (W/kg)^2'); ax.set_title(title); ax.legend(fontsize=8)
    fig.suptitle(f'{model_name} / Prosthetic / Exact Windowing — train vs val — '
                 f'FINAL ({len(epochs)}/{cfg.total_epochs} epochs)')
    M.plt.tight_layout()
    out_png = os.path.join(ROOT, f'{model_name.lower()}_prosthetic_exact_100ep_curve.png')
    M.plt.savefig(out_png, dpi=120)
    print(f'wrote {out_png}')
    # ---- end plot code ----

    ckpt = os.path.join(cfg.work_dir, f'epoch_{best_ep}.pth')
    model = M.build_model(cfg.model).to(M.DEVICE).eval()
    load_checkpoint(model, ckpt, map_location='cpu')

    rows = []
    for split in ['train', 'val', 'test']:
        p, y = M.run_inference(model, M.build_eval_loader(cfg, split))
        rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **M.metrics(p, y)))
    report_df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, f'{model_name.lower()}_prosthetic_exact_100ep_report.csv')
    report_df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}\n')
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(report_df.to_string(index=False))
    print()


if __name__ == '__main__':
    report('STGCN', os.path.join(ROOT, 'work_dirs/stgcn/prosthetic_wattkg/exact_100ep'))
    report('CTRGCN', os.path.join(ROOT, 'work_dirs/ctrgcn/prosthetic_wattkg/exact_100ep'))
