"""Train-vs-validation curve + best-epoch train/val/test report for the COMBINED runs
(able-bodied + prosthetic, STGCN and CTR-GCN), scored overall AND separately per population.

    PYTHONPATH=. python report_combined.py

The per-population rows are the ones to compare with the single-population models: the
combined test set is exactly the able-bodied test set plus the prosthetic test set.
Loads the actual config each run used, so a report can never drift out of sync with training.
"""
import glob
import os
import shutil

import sys

import numpy as np
import pandas as pd

try:
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from pyskl.models import build_model

    import gcn_eval as ge
except ImportError as e:
    sys.exit(f'{e}\nRun this with the training environment:\n'
             '  conda activate pyskl_310 && PYTHONPATH=. python report_combined.py')

ROOT = '/home/students/mmasabo1/summer26Research'
RESULTS = os.path.join(ROOT, 'Gait Metabolic Results', 'Combined')
RUNS = [('STGCN', 'configs/stgcn/combined_wattkg/exact_100ep.py'),
        ('CTRGCN', 'configs/ctrgcn/combined_wattkg/exact_100ep.py')]
GROUPS = ('All', 'able-bodied', 'prosthetic')


def report(model_name, config_path):
    cfg = Config.fromfile(os.path.join(ROOT, config_path))
    work_dir = os.path.join(ROOT, cfg.work_dir)   # configs store it relative to the project folder
    logs = glob.glob(os.path.join(work_dir, '*.log'))
    if not logs:
        sys.exit(f'No training log found in {work_dir} - has this run been trained?')
    tr = ge.epoch_train_mse(logs)
    vmse, vmae = ge.epoch_val_metrics(logs)
    best_ep = min(vmae, key=vmae.get)

    tag = f'{model_name.lower()}_combined_100ep'
    curve = os.path.join(ROOT, f'{tag}_curve.png')
    ge.plot_train_vs_val(tr, vmse, vmae, best_ep, cfg.total_epochs,
                         f'{model_name} / Combined (able-bodied + prosthetic) / Exact Windowing', curve)

    model = build_model(cfg.model).to(ge.DEVICE).eval()
    load_checkpoint(model, os.path.join(work_dir, f'epoch_{best_ep}.pth'), map_location='cpu')

    rows = []
    for split in ['train', 'val', 'test']:
        loader = ge.build_eval_loader(cfg, split)
        p, y = ge.run_inference(model, loader)
        pops = np.array([v['population'] for v in loader.dataset.video_infos])
        assert len(pops) == len(y), 'predictions and clips are out of step'
        for g in GROUPS:
            keep = np.ones(len(y), bool) if g == 'All' else pops == g
            rows.append(dict(Split=split, Population=g, Epoch=best_ep, N=int(keep.sum()), **ge.metrics(p[keep], y[keep])))
    df = pd.DataFrame(rows)
    out_csv = os.path.join(ROOT, f'{tag}_report.csv')
    df.to_csv(out_csv, index=False)
    print(f'wrote {out_csv}\n')
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(df.to_string(index=False))
    print()

    folder = os.path.join(RESULTS, model_name)
    os.makedirs(folder, exist_ok=True)
    shutil.copy(out_csv, folder)
    shutil.copy(curve, folder)
    shutil.copy(curve, os.path.join(folder, f'{model_name} Combined Training Curve.png'))
    return df


def sheet_rows(model_name, df):
    """Rows in the team sheet's layout: one per population scored by this combined model."""
    out = []
    for g in GROUPS:
        r = {s: df[(df.Split == s) & (df.Population == g)].iloc[0] for s in ('train', 'val', 'test')}
        label = {'All': 'Combined (all)', 'able-bodied': 'Combined (able-bodied)', 'prosthetic': 'Combined (prosthetic)'}[g]
        out.append(dict(Population=label, Model=model_name, Method='Exact Windowing', Units='W/kg',
                        BestEpoch=int(r['test'].Epoch), train_MAE=r['train'].MAE, train_MSE=r['train'].MSE,
                        val_MAE=r['val'].MAE, val_MSE=r['val'].MSE, test_MAE=r['test'].MAE, test_MSE=r['test'].MSE,
                        test_r=r['test'].r, test_R2=r['test'].R2, test_MRE=r['test'].MRE))
    return out


if __name__ == '__main__':
    rows = []
    for name, path in RUNS:
        rows += sheet_rows(name, report(name, path))
    summary = pd.DataFrame(rows).round(4)
    summary.to_csv(os.path.join(RESULTS, 'combined_results_summary.csv'), index=False)
    print('Combined tab rows (tab-separated, paste into Excel):\n')
    print(summary.to_csv(sep='\t', index=False))
