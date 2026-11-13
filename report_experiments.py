"""Best-epoch AND final-epoch report for the original runs and the 11/8 regularization
experiments (learning rate 0.01 -> 0.001, dropout 0 -> 0.5 / 0.7), for able-bodied and
prosthetic data, STGCN and CTR-GCN.

    PYTHONPATH=. ~/miniconda3/envs/pyskl_310/bin/python report_experiments.py [filter]
    (filter, optional: only runs whose config path contains it, e.g. `prosthetic` or `do05`;
     the summary files are only rewritten when no filter is given)

For every finished run it writes, into Gait Metabolic Results/<population folder>/<model>/:
    <MODEL> <Population> LR<lr> DR<dropout> Training Curve.png   train vs val, best AND final epoch marked
    <MODEL> <Population> LR<lr> DR<dropout> report.csv            train/val/test metrics at both checkpoints
and into Gait Metabolic Results/Regularization Experiments/:
    experiments_summary.csv           one row per run x checkpoint (best val epoch, final epoch)
    experiments_table.md              the compact table for Dr. Ramirez
    Validation MAE Comparison.png     original vs dropout 0.5 vs 0.7, per population and model

Metrics are computed exactly like the earlier reports (gcn_eval: one deterministic 100-frame
clip per sample), so the original-run best-epoch rows reproduce the numbers in the team sheet.
pyskl's own --test-best/--test-last numbers in the training logs average 10 clips per sample
and therefore differ slightly. LR and dropout are read from each run's own config.
"""
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

try:
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from pyskl.models import build_model

    import gcn_eval as ge
except ImportError as e:
    sys.exit(f'{e}\nRun this with the training environment:\n'
             '  PYTHONPATH=. ~/miniconda3/envs/pyskl_310/bin/python report_experiments.py')

ROOT = '/home/students/mmasabo1/summer26Research'
RESULTS = os.path.join(ROOT, 'Gait Metabolic Results')
SUMMARY_DIR = os.path.join(RESULTS, 'Regularization Experiments')

POPULATIONS = {   # name: (results folder, config pattern, long description)
    'Prosthetic': ('Prosthetic Only', 'configs/{m}/prosthetic_wattkg/merged_100ep{v}.py',
                   'Prosthetic (TAMUSA + Thailand)'),
    'Able-Bodied': ('Able Body Only', 'configs/{m}/ablebody2_wattkg/exact_100ep{v}.py', 'Able-Bodied'),
}
MODELS = ['STGCN', 'CTRGCN']
VARIANTS = ['', '_lr0001_do05', '_lr0001_do07']   # '' = the original (default) run
# first three slots of the validated categorical palette (pass all-pairs CVD checks)
COLORS = {'': '#2a78d6', '_lr0001_do05': '#eb6834', '_lr0001_do07': '#1baf7a'}


def settings(cfg):
    lr = cfg.optimizer['lr']
    do = cfg.model['cls_head'].get('dropout', 0.0) or 0.0
    return lr, do


def run_label(lr, do, variant):
    return 'Original (LR 0.01, no dropout)' if variant == '' else f'LR {lr:g} + dropout {do:g}'


def read_logs(work_dir):
    logs = glob.glob(os.path.join(work_dir, '*.log'))
    tr = ge.epoch_train_mse(logs)
    vmse, vmae = ge.epoch_val_metrics(logs)
    return tr, vmse, vmae


def report(pop, model_name, variant):
    folder, pattern, pop_desc = POPULATIONS[pop]
    config_path = pattern.format(m=model_name.lower(), v=variant)
    cfg = Config.fromfile(os.path.join(ROOT, config_path))
    work_dir = os.path.join(ROOT, cfg.work_dir)
    final_ep = cfg.total_epochs
    tr, vmse, vmae = read_logs(work_dir)
    if final_ep not in vmae or not os.path.exists(os.path.join(work_dir, f'epoch_{final_ep}.pth')):
        done = max(vmae) if vmae else 0
        print(f'-- {model_name} {pop}{variant}: not finished ({done}/{final_ep} epochs), skipped')
        return None
    best_ep = min(vmae, key=vmae.get)
    lr, do = settings(cfg)
    label = run_label(lr, do, variant)
    name = f'{model_name} {pop} LR{lr:g} DR{do:g}'

    out_dir = os.path.join(RESULTS, folder, model_name)
    os.makedirs(out_dir, exist_ok=True)
    ge.plot_train_vs_val(
        tr, vmse, vmae, best_ep, final_ep,
        f'{model_name} / {pop_desc} / {label}',
        os.path.join(out_dir, f'{name} Training Curve.png'), final_ep=final_ep,
        train_label='train MSE (mean batch loss, dropout active)' if do else 'train MSE (mean batch loss)')

    model = build_model(cfg.model).to(ge.DEVICE).eval()
    rows = []
    for which, ep in [('best val', best_ep), ('final', final_ep)]:
        load_checkpoint(model, os.path.join(work_dir, f'epoch_{ep}.pth'), map_location='cpu')
        for split in ['train', 'val', 'test']:
            p, y = ge.run_inference(model, ge.build_eval_loader(cfg, split))
            rows.append(dict(Checkpoint=which, Epoch=ep, Split=split, N=len(y), **ge.metrics(p, y)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f'{name} report.csv'), index=False)

    out = []
    for which in ['best val', 'final']:
        r = {s: df[(df.Checkpoint == which) & (df.Split == s)].iloc[0] for s in ('train', 'val', 'test')}
        out.append(dict(Population=pop, Model=model_name, Run=label, LR=lr, Dropout=do,
                        Checkpoint=which, Epoch=int(r['test'].Epoch),
                        train_MAE=r['train'].MAE, val_MAE=r['val'].MAE,
                        test_MAE=r['test'].MAE, test_MSE=r['test'].MSE, test_RMSE=r['test'].RMSE,
                        test_MRE_pct=100 * r['test'].MRE, test_r=r['test'].r, test_R2=r['test'].R2))
    print(f'== {name}: best epoch {best_ep}, test MAE best {out[0]["test_MAE"]:.4f} / '
          f'final {out[1]["test_MAE"]:.4f}')
    return out


def comparison_plot(out_png):
    """Validation MAE per epoch: original vs the two regularized runs, one panel per population x model."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
    for i, pop in enumerate(POPULATIONS):
        for j, model_name in enumerate(MODELS):
            ax = axes[i, j]
            folder, pattern, pop_desc = POPULATIONS[pop]
            for v in VARIANTS:
                cfg = Config.fromfile(os.path.join(ROOT, pattern.format(m=model_name.lower(), v=v)))
                _, _, vmae = read_logs(os.path.join(ROOT, cfg.work_dir))
                if not vmae:
                    continue
                lr, do = settings(cfg)
                es = sorted(vmae)
                best = min(vmae, key=vmae.get)
                ax.plot(es, [vmae[e] for e in es], color=COLORS[v], lw=2,
                        label=f'{run_label(lr, do, v)}: best {vmae[best]:.3f} @ {best}, '
                              f'final {vmae[es[-1]]:.3f}' + ('' if es[-1] == cfg.total_epochs else ' (running)'))
                ax.plot([best], [vmae[best]], 'o', ms=8, color=COLORS[v], mec='white', mew=2)
            ax.set_title(f'{model_name} / {pop_desc}', fontsize=11)
            ax.set_ylim(top=min(ax.get_ylim()[1], 0.8 if pop == 'Able-Bodied' else 0.45))
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, frameon=False)
            if i == 1:
                ax.set_xlabel('epoch')
            if j == 0:
                ax.set_ylabel('validation MAE (W/kg)')
    fig.suptitle('Regularization experiments: validation MAE per epoch (dot = best epoch)', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f'wrote {out_png}')


def ramirez_table(summary):
    """One line per run: best-epoch vs final-epoch test results side by side."""
    lines = ['| Population | Model | Settings | Best epoch | Test MAE (best) | Test MAE (final, ep 100) '
             '| Test r (best) | Test r (final) |',
             '|---|---|---|---|---|---|---|---|']
    for (pop, model_name, run), g in summary.groupby(['Population', 'Model', 'Run'], sort=False):
        b = g[g.Checkpoint == 'best val'].iloc[0]
        f = g[g.Checkpoint == 'final'].iloc[0]
        lines.append(f'| {pop} | {model_name} | {run} | {b.Epoch} | {b.test_MAE:.4f} | {f.test_MAE:.4f} '
                     f'| {b.test_r:.4f} | {f.test_r:.4f} |')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    wanted = sys.argv[1] if len(sys.argv) > 1 else ''
    rows = []
    for pop, (_, pattern, _) in POPULATIONS.items():
        for model_name in MODELS:
            for v in VARIANTS:
                if wanted in pattern.format(m=model_name.lower(), v=v):
                    rows += report(pop, model_name, v) or []
    if wanted:
        sys.exit(0)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(os.path.join(SUMMARY_DIR, 'experiments_summary.csv'), index=False)
    with open(os.path.join(SUMMARY_DIR, 'experiments_table.md'), 'w') as fh:
        fh.write(ramirez_table(summary))
    comparison_plot(os.path.join(SUMMARY_DIR, 'Validation MAE Comparison.png'))
    print('\n' + ramirez_table(summary))
