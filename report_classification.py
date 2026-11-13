"""Training curve + best-epoch train/val/test report for the able-bodied vs prosthetic
CLASSIFICATION runs (STGCN and CTR-GCN, population_cls/able_vs_prosthetic.pkl).

    PYTHONPATH=. python report_classification.py

Best epoch = highest validation mean class accuracy (balanced accuracy), the checkpoint pyskl saved
as best_mean_class_accuracy_epoch_N.pth. Train/val/test are all scored from that one checkpoint, and
test accuracy is also broken down by site (AbleBody2 / TAMUSA / Thailand).
Loads the actual config each run used, so a report can never drift out of sync with training.
"""
import glob
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from pyskl.models import build_model

    import gcn_eval as ge
except ImportError as e:
    sys.exit(f'{e}\nRun this with the training environment:\n'
             '  conda activate pyskl_310 && PYTHONPATH=. python report_classification.py')

ROOT = '/home/students/mmasabo1/summer26Research'
RESULTS = os.path.join(ROOT, 'Gait Metabolic Results', 'Classification')
RUNS = [('STGCN', 'configs/stgcn/population_cls/able_vs_prosthetic_100ep.py'),
        ('CTRGCN', 'configs/ctrgcn/population_cls/able_vs_prosthetic_100ep.py')]
CLASSES = ('able-bodied', 'prosthetic')   # label = index


def parse_log(log_paths):
    """Per-epoch mean train loss / accuracy (from the batch lines) and val accuracy / balanced accuracy."""
    loss, acc, val_acc, val_bal = {}, {}, {}, {}
    for lp in sorted(log_paths):
        for line in open(lp):
            m = re.search(r'Epoch \[(\d+)\]\[.*top1_acc: ([0-9.]+).*loss_cls: ([0-9.]+)', line)
            if m:
                e = int(m.group(1))
                acc.setdefault(e, []).append(float(m.group(2)))
                loss.setdefault(e, []).append(float(m.group(3)))
            m = re.search(r'Epoch\(val\) \[(\d+)\].*top1_acc: ([0-9.]+), mean_class_accuracy: ([0-9.]+)', line)
            if m:
                e = int(m.group(1))
                val_acc[e], val_bal[e] = float(m.group(2)), float(m.group(3))
    mean = lambda d: {e: float(np.mean(v)) for e, v in d.items()}
    return mean(loss), mean(acc), val_acc, val_bal


def auc(y, score):
    """ROC AUC as the chance a random prosthetic clip scores above a random able-bodied one (ties = 1/2)."""
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return float('nan')
    order = np.argsort(np.concatenate([pos, neg]), kind='mergesort')
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):                      # average ranks over ties
        k = allv == v
        ranks[k] = ranks[k].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@torch.no_grad()
def predict(model, loader):
    """Softmax probability of 'prosthetic' for every clip, plus the true labels."""
    probs, labels = [], []
    for batch in loader:
        y = batch['label'].numpy().reshape(-1)
        out = np.asarray(model(batch['keypoint'].to(ge.DEVICE), return_loss=False)).reshape(len(y), -1)
        probs.append(out[:, 1]); labels.append(y)
    return np.concatenate(probs), np.concatenate(labels).astype(int)


def cls_metrics(prob, y):
    pred = (prob >= 0.5).astype(int)
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    fn, tp = int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())
    rec_able = tn / max(tn + fp, 1)
    rec_pros = tp / max(tp + fn, 1)
    prec_pros = tp / max(tp + fp, 1)
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=(rec_able + rec_pros) / 2,
                recall_able=rec_able, recall_prosthetic=rec_pros, precision_prosthetic=prec_pros,
                F1_prosthetic=2 * prec_pros * rec_pros / max(prec_pros + rec_pros, 1e-12),
                AUC=auc(y, prob),
                TN=tn, FP=fp, FN=fn, TP=tp)


def plot(model_name, loss, acc, val_acc, val_bal, best_ep, total_epochs, test, out_png):
    epochs = sorted(set(loss) & set(val_acc))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw=dict(width_ratios=[1, 1, 0.75]))
    ax = axes[0]
    ax.plot(epochs, [loss[e] for e in epochs], 'o-', ms=3, label='train loss (mean batch cross-entropy)')
    ax.axvline(best_ep, color='r', ls='--', lw=1, label=f'best epoch {best_ep}')
    ax.set_xlabel('epoch'); ax.set_ylabel('cross-entropy loss'); ax.set_title('training loss'); ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(epochs, [100 * acc[e] for e in epochs], 'o-', ms=3, label='train accuracy (mean over batches)')
    ax.plot(epochs, [100 * val_acc[e] for e in epochs], 's-', ms=3, label='val accuracy')
    ax.axvline(best_ep, color='r', ls='--', lw=1,
               label=f'best epoch {best_ep} (val balanced acc {100 * val_bal[best_ep]:.2f}%)')
    ax.set_xlabel('epoch'); ax.set_ylabel('accuracy (%)'); ax.set_title('train vs val accuracy'); ax.legend(fontsize=8)
    ax = axes[2]
    cm = np.array([[test['TN'], test['FP']], [test['FN'], test['TP']]])
    share = cm / cm.sum(1, keepdims=True)   # shade by share of each true class (classes are 582 vs 295)
    ax.imshow(share, cmap='Blues', vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i, j]}\n({100 * share[i, j]:.1f}%)', ha='center', va='center', fontsize=11,
                    color='white' if share[i, j] > 0.5 else 'black')
    ax.set_xticks([0, 1]); ax.set_xticklabels([f'predicted\n{c}' for c in CLASSES])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f'true\n{c}' for c in CLASSES])
    ax.set_title(f"test set, epoch {best_ep}: accuracy {100 * test['accuracy']:.2f}%")
    fig.suptitle(f'{model_name} / able-bodied vs prosthetic classification — FINAL ({len(epochs)}/{total_epochs} epochs)')
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'wrote {out_png}')


def report(model_name, config_path):
    cfg = Config.fromfile(os.path.join(ROOT, config_path))
    work_dir = os.path.join(ROOT, cfg.work_dir)   # configs store it relative to the project folder
    logs = glob.glob(os.path.join(work_dir, '*.log'))
    best = glob.glob(os.path.join(work_dir, 'best_mean_class_accuracy_epoch_*.pth'))
    if not logs or not best:
        sys.exit(f'No training log / best checkpoint in {work_dir} - has this run been trained?')
    loss, acc, val_acc, val_bal = parse_log(logs)
    best_ep = int(re.search(r'epoch_(\d+)\.pth', best[0]).group(1))

    model = build_model(cfg.model).to(ge.DEVICE).eval()
    load_checkpoint(model, best[0], map_location='cpu')

    rows, site_rows = [], []
    for split in ['train', 'val', 'test']:
        loader = ge.build_eval_loader(cfg, split)
        prob, y = predict(model, loader)
        assert len(prob) == len(loader.dataset.video_infos), 'predictions and clips are out of step'
        rows.append(dict(Split=split, Epoch=best_ep, N=len(y), **cls_metrics(prob, y)))
        if split == 'test':
            sites = np.array([v['source'] for v in loader.dataset.video_infos])
            for s in sorted(set(sites)):
                k = sites == s
                site_rows.append(dict(Split='test', Site=s, N=int(k.sum()),
                                      accuracy=float(((prob[k] >= 0.5).astype(int) == y[k]).mean())))
    df = pd.DataFrame(rows)
    tag = f'{model_name.lower()}_population_cls_100ep'
    out_csv = os.path.join(ROOT, f'{tag}_report.csv')
    df.to_csv(out_csv, index=False)
    sites = pd.DataFrame(site_rows)
    site_csv = os.path.join(ROOT, f'{tag}_test_by_site.csv')
    sites.to_csv(site_csv, index=False)
    with pd.option_context('display.float_format', lambda x: f'{x:.4f}'):
        print(df.to_string(index=False)); print(); print(sites.to_string(index=False)); print()

    curve = os.path.join(ROOT, f'{tag}_curve.png')
    plot(model_name, loss, acc, val_acc, val_bal, best_ep, cfg.total_epochs, rows[-1], curve)

    folder = os.path.join(RESULTS, model_name)
    os.makedirs(folder, exist_ok=True)
    for f in (out_csv, site_csv, curve):
        shutil.copy(f, folder)
    shutil.copy(curve, os.path.join(folder, f'{model_name} Classification Training Curve.png'))
    return df


def sheet_row(model_name, df):
    r = {s: df[df.Split == s].iloc[0] for s in ('train', 'val', 'test')}
    return dict(Task='Able-bodied vs prosthetic', Model=model_name, BestEpoch=int(r['test'].Epoch),
                train_acc=r['train'].accuracy, val_acc=r['val'].accuracy, val_balanced_acc=r['val'].balanced_accuracy,
                test_acc=r['test'].accuracy, test_balanced_acc=r['test'].balanced_accuracy,
                test_recall_able=r['test'].recall_able, test_recall_prosthetic=r['test'].recall_prosthetic,
                test_F1_prosthetic=r['test'].F1_prosthetic, test_AUC=r['test'].AUC)


if __name__ == '__main__':
    rows = [sheet_row(name, report(name, path)) for name, path in RUNS]
    summary = pd.DataFrame(rows).round(4)
    os.makedirs(RESULTS, exist_ok=True)
    summary.to_csv(os.path.join(RESULTS, 'classification_results_summary.csv'), index=False)
    print('Classification rows (tab-separated, paste into Excel):\n')
    print(summary.to_csv(sep='\t', index=False))
