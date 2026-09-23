"""Train-vs-validation curve + best-epoch train/val/test report for the able-bodied
STGCN run, trained via the regular pipeline:
    bash train_stgcn_exact_100ep.sh
    (= tools/dist_train.sh configs/stgcn/ablebody2_wattkg/exact_100ep.py 1 ...)

Loads the *actual* config file that run used, so this can never drift out of sync with
what was really trained.
"""
import glob
import os

import pandas as pd
from mmcv import Config
from mmcv.runner import load_checkpoint
from pyskl.models import build_model

import gcn_eval as ge

ROOT = '/home/students/mmasabo1/summer26Research'
CONFIG = os.path.join(ROOT, 'configs/stgcn/ablebody2_wattkg/exact_100ep.py')

cfg = Config.fromfile(CONFIG)
logs = glob.glob(os.path.join(cfg.work_dir, '*.log'))
tr = ge.epoch_train_mse(logs)
vmse, vmae = ge.epoch_val_metrics(logs)
best_ep = min(vmae, key=vmae.get)

ge.plot_train_vs_val(tr, vmse, vmae, best_ep, cfg.total_epochs, 'STGCN / Exact Windowing',
                      os.path.join(ROOT, 'stgcn_exact_100ep_curve.png'))

ckpt = os.path.join(cfg.work_dir, f'epoch_{best_ep}.pth')
model = build_model(cfg.model).to(ge.DEVICE).eval()
load_checkpoint(model, ckpt, map_location='cpu')

rows = []
for split in ['train', 'val', 'test']:
    p, y = ge.run_inference(model, ge.build_eval_loader(cfg, split))
    rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **ge.metrics(p, y)))
report = pd.DataFrame(rows)
out_csv = os.path.join(ROOT, 'stgcn_exact_100ep_report.csv')
report.to_csv(out_csv, index=False)
print(f'wrote {out_csv}\n')
with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
    print(report.to_string(index=False))
