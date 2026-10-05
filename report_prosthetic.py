"""Train-vs-validation curve + best-epoch train/val/test report for the prosthetic runs
(STGCN and CTR-GCN, Exact Windowing), all trained via the regular pipeline:
    TAMUSA only:        train_{stgcn,ctrgcn}_prosthetic_100ep.sh         (prosthetic_wattkg/exact.pkl)
    TAMUSA + Thailand:  train_{stgcn,ctrgcn}_prosthetic_merged_100ep.sh  (prosthetic_wattkg/exact_merged.pkl)

Loads the *actual* config file each run used, so a report can never drift out of sync
with what was really trained.
"""
import glob
import os
import sys

import pandas as pd
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.models import build_model

import gcn_eval as ge

ROOT = '/home/students/mmasabo1/summer26Research'
# (model, config, output tag, plot title)
RUNS = [
    ('STGCN', 'configs/stgcn/prosthetic_wattkg/exact_100ep.py', 'exact_100ep', 'TAMUSA only'),
    ('CTRGCN', 'configs/ctrgcn/prosthetic_wattkg/exact_100ep.py', 'exact_100ep', 'TAMUSA only'),
    ('STGCN', 'configs/stgcn/prosthetic_wattkg/merged_100ep.py', 'merged_100ep', 'TAMUSA + Thailand'),
    ('CTRGCN', 'configs/ctrgcn/prosthetic_wattkg/merged_100ep.py', 'merged_100ep', 'TAMUSA + Thailand'),
]


def report(model_name, config_path, tag, cohort):
    cfg = Config.fromfile(os.path.join(ROOT, config_path))
    logs = glob.glob(os.path.join(cfg.work_dir, '*.log'))
    tr = ge.epoch_train_mse(logs)
    vmse, vmae = ge.epoch_val_metrics(logs)
    best_ep = min(vmae, key=vmae.get)

    ge.plot_train_vs_val(tr, vmse, vmae, best_ep, cfg.total_epochs,
                          f'{model_name} / Prosthetic ({cohort}) / Exact Windowing',
                          os.path.join(ROOT, f'{model_name.lower()}_prosthetic_{tag}_curve.png'))

    ckpt = os.path.join(cfg.work_dir, f'epoch_{best_ep}.pth')
    model = build_model(cfg.model).to(ge.DEVICE).eval()
    load_checkpoint(model, ckpt, map_location='cpu')

    rows = []
    for split in ['train', 'val', 'test']:
        p, y = ge.run_inference(model, ge.build_eval_loader(cfg, split))
        rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **ge.metrics(p, y)))
    report_df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, f'{model_name.lower()}_prosthetic_{tag}_report.csv')
    report_df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}\n')
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(report_df.to_string(index=False))
    print()


if __name__ == '__main__':
    # optional filter, e.g. `python report_prosthetic.py merged` runs only the merged-cohort reports
    wanted = sys.argv[1] if len(sys.argv) > 1 else ''
    for model_name, config_path, tag, cohort in RUNS:
        if wanted in tag:
            report(model_name, config_path, tag, cohort)
