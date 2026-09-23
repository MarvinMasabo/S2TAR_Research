"""Build the three repo-side deliverables from the transcript tasks:

  1. paper_table_test.{csv,md}  - TEST-split results, all metrics, per model x method (W/kg + Watts)
  2. overfitting_check.{png,csv} - train vs val MSE across epochs for all 6 runs + verdict
  3. demographics_ablebody.{csv,md} - able-bodied participant demographics (age/height/body mass)
"""
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/home/students/mmasabo1/summer26Research'
METHODS = ['nearest', 'exact', 'windowing']
METHOD_LABEL = {'nearest': 'Nearest EE', 'exact': 'Exact Windowing', 'windowing': 'Windowing Avg Drop'}
METHOD_SHORT = {'nearest': 'nearest', 'exact': 'exact', 'windowing': 'dropped'}
MODELS = ['STGCN', 'PoseC3D']
WORKDIR = {'STGCN': 'work_dirs/stgcn/ablebody2_wattkg', 'PoseC3D': 'work_dirs/posec3d/ablebody2_wattkg'}


# ------------------------------------------------------------------ 1. paper table
def paper_table():
    src = pd.read_csv(os.path.join(ROOT, 'evaluation_results_trained.csv'))
    # already test-split, both units. Just order + relabel for the paper.
    order = {'Nearest EE': 0, 'Exact Windowing': 1, 'Windowing Avg Drop': 2}
    src['mo'] = src['Model'].map({'STGCN': 0, 'PoseC3D': 1})
    src['me'] = src['Experiment'].map(order)
    src['un'] = src['Units'].map({'W/kg': 0, 'Watts': 1})
    src = src.sort_values(['mo', 'me', 'un']).drop(columns=['mo', 'me', 'un'])
    src = src.rename(columns={'Experiment': 'Method', 'Epoch': 'BestEpoch', 'TestSamples': 'N',
                              'MAE': 'Test_MAE', 'MSE': 'Test_MSE', 'RMSE': 'Test_RMSE',
                              'Pearson r': 'Test_PearsonR', 'R2': 'Test_R2'})
    src.to_csv(os.path.join(ROOT, 'paper_table_test.csv'), index=False)

    with open(os.path.join(ROOT, 'paper_table_test.md'), 'w') as f:
        f.write('# Test-split results  (best checkpoint by val MAE)\n\n')
        for unit in ['W/kg', 'Watts']:
            sub = src[src['Units'] == unit]
            f.write(f'## Units: {unit}\n\n')
            f.write('| Model | Method | Best epoch | N | Test MSE | Test MAE | Test RMSE | Test Pearson r | Test R2 |\n')
            f.write('|---|---|--:|--:|--:|--:|--:|--:|--:|\n')
            for _, r in sub.iterrows():
                f.write(f"| {r.Model} | {r.Method} | {int(r.BestEpoch)} | {int(r.N)} | "
                        f"{r.Test_MSE:.4f} | {r.Test_MAE:.4f} | {r.Test_RMSE:.4f} | "
                        f"{r.Test_PearsonR:.4f} | {r.Test_R2:.4f} |\n")
            f.write('\n')
    print('[1] wrote paper_table_test.csv / .md')
    return src


# ------------------------------------------------------------------ 2. overfitting
def epoch_train_mse(log_paths):
    """Mean batch loss_cls per epoch (MSELoss weight 1.0 -> batch MSE in (W/kg)^2)."""
    per = {}
    for lp in sorted(log_paths):
        for line in open(lp):
            m = re.search(r'Epoch \[(\d+)\]\[.*loss_cls: ([0-9.]+)', line)
            if m:
                per.setdefault(int(m.group(1)), []).append(float(m.group(2)))
    return {e: float(np.mean(v)) for e, v in per.items()}


def epoch_val_metrics(log_paths):
    """pyskl regression logs val MSE in the 'mean_class_accuracy' slot."""
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


def overfitting_check():
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), squeeze=False)
    for i, model in enumerate(MODELS):
        for j, method in enumerate(METHODS):
            d = os.path.join(ROOT, WORKDIR[model], method)
            logs = glob.glob(os.path.join(d, '*.log'))
            tr = epoch_train_mse(logs)
            vmse, vmae = epoch_val_metrics(logs)
            es = sorted(set(tr) & set(vmse))
            t = [tr[e] for e in es]
            v = [vmse[e] for e in es]
            best_e = min(vmae, key=vmae.get) if vmae else None

            ax = axes[i][j]
            ax.plot(es, t, 'o-', ms=3, label='train MSE (mean loss)')
            ax.plot(es, v, 's-', ms=3, label='val MSE')
            if best_e:
                ax.axvline(best_e, color='r', ls='--', lw=1, label=f'best epoch {best_e}')
            # scale the y-axis off epoch>=2 (epoch 1's huge starting loss would otherwise
            # squash the whole rest of the curve into an unreadable band near the bottom);
            # epoch 1's point still plots, it just runs off the top of the visible frame
            visible = [x for e, x in zip(es, t) if e >= 2] + [x for e, x in zip(es, v) if e >= 2]
            if visible:
                ax.set_ylim(0, max(visible) * 1.15)
            ax.set_title(f'{model} / {METHOD_SHORT[method]}' + (' (epoch 1 off-scale)' if es and es[0] == 1 else ''))
            ax.set_xlabel('epoch'); ax.set_ylabel('MSE (W/kg)^2'); ax.legend(fontsize=8)

            last = es[-1]
            gap = v[-1] - t[-1]
            vmin_e = min(vmse, key=vmse.get)
            # judge on the training tail (last third), not the whole descending curve
            tail = [e for e in es if e >= last - max(3, last // 3)]
            tail_v = np.array([vmse[e] for e in tail])
            tail_t = np.array([tr[e] for e in tail])
            tail_cv = tail_v.std() / tail_v.mean()               # tail volatility
            tail_gap = float((tail_v - tail_t).mean())            # mean val-train gap on tail
            rising = vmse[last] > 1.15 * vmse[vmin_e] and vmin_e < 0.7 * last
            verdict = ('OVERFIT'  if (rising and tail_gap > 0.15) else
                       'UNSTABLE' if tail_cv > 0.20 else
                       'OK')
            rows.append(dict(Model=model, Method=METHOD_SHORT[method], Epochs=last,
                             best_epoch=best_e, val_MSE_min=round(vmse[vmin_e], 4),
                             val_MSE_min_epoch=vmin_e, val_MSE_final=round(vmse[last], 4),
                             train_MSE_final=round(tr[last], 4),
                             tail_gap=round(tail_gap, 4), tail_cv=round(tail_cv, 3),
                             verdict=verdict))
    plt.tight_layout()
    plt.savefig(os.path.join(ROOT, 'overfitting_check.png'), dpi=110)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, 'overfitting_check.csv'), index=False)
    print('[2] wrote overfitting_check.png / .csv')
    print(df.to_string(index=False))
    return df


# ------------------------------------------------------------------ 3. demographics
def demographics():
    d = pickle.load(open(os.path.join(ROOT, 'AbleBody2_full.pkl'), 'rb'))
    by = {}
    for a in d['annotations']:
        by.setdefault(a['name'], a)           # one record per participant
    P = list(by.values())
    sex = {'M': sum(p['gender'] == 1.0 for p in P), 'F': sum(p['gender'] == 0.0 for p in P)}

    def stat(field, label, unit):
        v = np.array([p[field] for p in P], float)
        return dict(Variable=label, Unit=unit, n=len(v),
                    Mean=round(v.mean(), 1), SD=round(v.std(ddof=1), 1),
                    Min=round(v.min(), 1), Max=round(v.max(), 1),
                    Range=f'{v.min():.1f}-{v.max():.1f}')

    tbl = pd.DataFrame([
        stat('age', 'Age', 'years'),
        stat('height', 'Height', 'cm'),
        stat('weight', 'Body mass', 'kg'),
    ])
    tbl.to_csv(os.path.join(ROOT, 'demographics_ablebody.csv'), index=False)

    import collections
    mods = collections.Counter(a['modality'] for a in d['annotations'])
    with open(os.path.join(ROOT, 'demographics_ablebody.md'), 'w') as f:
        f.write('# Able-bodied participant demographics (AbleBody2)\n\n')
        f.write(f'N = {len(P)} participants ({sex["M"]} male, {sex["F"]} female). '
                f'All able-bodied; no limb-side breakdown.\n\n')
        f.write('| Variable | Unit | n | Mean | SD | Range |\n|---|---|--:|--:|--:|--:|\n')
        for _, r in tbl.iterrows():
            f.write(f'| {r.Variable} | {r.Unit} | {r.n} | {r.Mean} | {r.SD} | {r.Range} |\n')
        f.write(f'\nClips per walking speed: '
                + ', '.join(f'{k} {v}' for k, v in sorted(mods.items())) + '.\n')
    print('[3] wrote demographics_ablebody.csv / .md')
    print(tbl.to_string(index=False))
    print('sex:', sex, '| modalities:', dict(mods))


if __name__ == '__main__':
    paper_table()
    print()
    overfitting_check()
    print()
    demographics()
