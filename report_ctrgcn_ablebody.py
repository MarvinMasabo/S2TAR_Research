"""Train-vs-validation curve + best-epoch train/val/test report for the real able-bodied
CTR-GCN run (work_dirs/ctrgcn/ablebody2_wattkg/exact_100ep, trained via
tools/dist_train.sh + configs/ctrgcn/ablebody2_wattkg/exact_100ep.py). Same plotting code
as train_and_plot_gcn.py's plot_and_report() / report_prosthetic.py -- this just points it
at the able-bodied CTR-GCN work_dir instead.
"""
import glob
import os

import pandas as pd
from mmcv.runner import load_checkpoint

import train_and_plot_gcn as M

ROOT = '/home/students/mmasabo1/summer26Research'

M.MODEL = 'CTRGCN'
M.ANN_FILE = os.path.join(ROOT, 'ablebody2_wattkg/exact.pkl')
M.WORK_DIR = os.path.join(ROOT, 'work_dirs/ctrgcn/ablebody2_wattkg/exact_100ep')
cfg = M.build_config()

logs = glob.glob(os.path.join(cfg.work_dir, '*.log'))
tr = M.epoch_train_mse(logs)
vmse, vmae = M.epoch_val_metrics(logs)
epochs = sorted(set(tr) & set(vmse))
best_ep = min(vmae, key=vmae.get)

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
fig.suptitle(f'CTRGCN / Exact Windowing — train vs val — FINAL ({len(epochs)}/{cfg.total_epochs} epochs)')
M.plt.tight_layout()
out_png = os.path.join(ROOT, 'ctrgcn_exact_100ep_curve.png')
M.plt.savefig(out_png, dpi=120)
print(f'wrote {out_png}')

ckpt = os.path.join(cfg.work_dir, f'epoch_{best_ep}.pth')
model = M.build_model(cfg.model).to(M.DEVICE).eval()
load_checkpoint(model, ckpt, map_location='cpu')

rows = []
for split in ['train', 'val', 'test']:
    p, y = M.run_inference(model, M.build_eval_loader(cfg, split))
    rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **M.metrics(p, y)))
report_df = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, 'ctrgcn_exact_100ep_report.csv')
report_df.to_csv(out_csv, index=False)
print(f'wrote {out_csv}\n')
with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
    print(report_df.to_string(index=False))
