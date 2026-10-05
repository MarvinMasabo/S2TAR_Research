"""Plain-language status report for pyskl training runs, read straight from their logs.

    python check_run.py                    # the combined STGCN + CTR-GCN runs
    python check_run.py CONFIG [CONFIG..]  # any runs, e.g. configs/stgcn/prosthetic_wattkg/merged_100ep.py
    python check_run.py --last 15          # show more epochs in the table (default 10)
    python check_run.py --plot             # also draw live_training_curves.png (needs pyskl_310)

For each run: running / finished / stopped, progress and time left, the latest validation
epochs, the best epoch so far, and a short diagnosis (improving, settling, overfitting, NaNs).
"""
import argparse
import datetime as dt
import glob
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT = ['configs/stgcn/combined_wattkg/exact_100ep.py',
           'configs/ctrgcn/combined_wattkg/exact_100ep.py']

TS = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
TRAIN = re.compile(TS + r'.*Epoch \[(\d+)\]\[(\d+)/(\d+)\]\s+lr: ([0-9.e+-]+), eta: ([^,]+),.*loss_cls: ([0-9.naNinf]+)')
VAL = re.compile(TS + r'.*Epoch\(val\) \[(\d+)\].*mean_class_accuracy: ([0-9.naNinf]+), mean_absolute_error: ([0-9.naNinf]+)')
TEST_HDR = re.compile(r'Testing results of the (best|last) checkpoint')
TEST_MAE = re.compile(r'mean_absolute_error: ([0-9.]+)')


def config_value(path, key):
    m = re.search(rf"^{key}\s*=\s*(.+)$", open(path).read(), re.M)
    return m.group(1).strip().strip("'\"") if m else None


def parse_log(work_dir):
    logs = sorted(glob.glob(os.path.join(work_dir, '*.log')))
    train, val, last_train, tests = {}, {}, None, {}
    pending = None
    for lp in logs:
        for line in open(lp, errors='replace'):
            m = TRAIN.search(line)
            if m:
                ts, ep, it, n, lr, eta, loss = m.groups()
                train.setdefault(int(ep), []).append(float(loss))
                last_train = dict(ts=ts, epoch=int(ep), it=int(it), n=int(n), lr=float(lr), eta=eta.strip())
                continue
            m = VAL.search(line)
            if m:
                ts, ep, mse, mae = m.groups()
                val[int(ep)] = dict(ts=dt.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), mse=float(mse), mae=float(mae))
                continue
            m = TEST_HDR.search(line)
            if m:
                pending = m.group(1)
                continue
            if pending:
                m = TEST_MAE.search(line)
                if m:
                    tests[pending] = float(m.group(1))
                    pending = None
    return logs, train, val, last_train, tests


def is_running(config):
    try:
        ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    except OSError:
        return False
    return any('tools/train.py' in l and config in l for l in ps.splitlines())


def mean(xs):
    return sum(xs) / len(xs) if xs else float('nan')


def diagnose(train, val, best_ep, last_ep):
    notes = []
    eps = sorted(val)
    if any(v['mse'] != v['mse'] for v in val.values()) or any(x != x for v in train.values() for x in v):
        notes.append('PROBLEM: NaN in the loss or validation - training has diverged; stop and lower the learning rate.')
        return notes
    if len(eps) < 6:
        notes.append('Too early to judge - wait for at least 6 validated epochs.')
        return notes
    last5 = [val[e]['mae'] for e in eps[-5:]]
    prev5 = [val[e]['mae'] for e in eps[-10:-5]] or last5
    change = (mean(last5) - mean(prev5)) / mean(prev5)
    tr = [mean(train[e]) for e in sorted(train) if e in val or e <= last_ep]
    tr_change = (mean(tr[-5:]) - mean(tr[-10:-5])) / mean(tr[-10:-5]) if len(tr) >= 10 else 0
    if change < -0.02:
        notes.append(f'Validation MAE is still improving ({100 * change:+.0f}% over the last 5 epochs vs the 5 before).')
    elif change > 0.02 and tr_change < -0.02:
        notes.append(f'Validation MAE is getting worse ({100 * change:+.0f}%) while training keeps improving - mild '
                     'overfitting. Normal late in training; the best epoch, not the last, is what gets reported.')
    elif change > 0.02:
        notes.append(f'Validation MAE rose {100 * change:+.0f}% over the last 5 epochs - watch the next few.')
    else:
        notes.append('Validation MAE has levelled off (within 2% over the last 5 epochs).')

    recent = [val[e]['mse'] for e in eps[-10:]]
    spread = (max(recent) - min(recent)) / sorted(recent)[len(recent) // 2]
    earlier = [val[e]['mse'] for e in eps[-20:-10]]
    if spread > 1:
        msg = 'Validation is bouncing a lot (last-10 range more than 2x its median)'
        if earlier and (max(earlier) - min(earlier)) > (max(recent) - min(recent)):
            msg += ', but less than the 10 epochs before - it is settling as the learning rate drops.'
        else:
            msg += '. Normal while the learning rate is high; if it persists late, try a 10x smaller rate.'
        notes.append(msg)
    elif spread > 0.3:
        notes.append('Validation has some epoch-to-epoch bounce - normal.')
    else:
        notes.append('Validation is steady from epoch to epoch.')

    above = (mean(last5) - val[best_ep]['mae']) / val[best_ep]['mae']
    if above > 0.05 and last_ep - best_ep >= 10:
        notes.append(f'Recent validation MAE is {100 * above:.0f}% above the best epoch\'s - the model has drifted past '
                     'its best (mild overfitting). The best epoch is the one that gets reported.')
    since = last_ep - best_ep
    if since >= 20:
        notes.append(f'No new best for {since} epochs - the best epoch ({best_ep}) is probably final.')
    gap = mean(tr[-3:]) / mean(recent[-3:]) if recent else 1
    if gap < 0.5:
        notes.append(f'Training error is {1 / gap:.1f}x lower than validation error - the model fits training data '
                     'much better than unseen data.')
    return notes


def report(config, n_last):
    path = os.path.join(ROOT, config)
    name = config.replace('configs/', '').replace('.py', '')
    print('=' * 78)
    print(f'  {name}')
    print('=' * 78)
    if not os.path.exists(path):
        print(f'  config not found: {config}\n')
        return
    work_dir = os.path.normpath(os.path.join(ROOT, config_value(path, 'work_dir')))
    total = int(config_value(path, 'total_epochs') or 0)
    logs, train, val, last, tests = parse_log(work_dir)
    if not logs:
        print(f'  no log yet in {os.path.relpath(work_dir, ROOT)}\n')
        return
    running = is_running(config)
    done_ep = max(val) if val else 0
    status = 'RUNNING' if running else ('FINISHED' if tests or done_ep >= total else 'STOPPED (not running, not finished)')
    print(f'  Status:   {status}')
    if last:
        frac = ((last['epoch'] - 1) + last['it'] / last['n']) / total if total else 0
        print(f'  Progress: epoch {last["epoch"]}/{total}, iteration {last["it"]}/{last["n"]}  ({100 * frac:.0f}%)')
        if running:
            print(f'  Time left (pyskl estimate): {last["eta"]}    learning rate now: {last["lr"]:.2e}')
        print(f'  Last log line: {last["ts"]}')
    eps = sorted(val)
    if len(eps) >= 2:
        per = (val[eps[-1]]['ts'] - val[eps[0]]['ts']).total_seconds() / (len(eps) - 1)
        print(f'  Speed:    {per / 60:.1f} min per epoch')
    if not val:
        print('  No validated epochs yet.\n')
        return
    best_ep = min(val, key=lambda e: val[e]['mae'])
    print(f'  Best so far: epoch {best_ep}, val MAE {val[best_ep]["mae"]:.4f} W/kg, val MSE {val[best_ep]["mse"]:.4f}'
          f'   ({done_ep - best_ep} epochs ago)')
    print(f'\n  {"epoch":>5}  {"train MSE":>9}  {"val MSE":>8}  {"val MAE":>8}')
    for e in eps[-n_last:]:
        tr = mean(train.get(e, []))
        mark = '  <- best' if e == best_ep else ''
        print(f'  {e:>5}  {tr:>9.4f}  {val[e]["mse"]:>8.4f}  {val[e]["mae"]:>8.4f}{mark}')
    print('\n  Diagnosis:')
    for note in diagnose(train, val, best_ep, done_ep):
        print(f'   - {note}')
    if tests:
        print('\n  pyskl end-of-run test MAE (10 samples per clip; use report_*.py for the reported numbers):')
        for k in ('best', 'last'):
            if k in tests:
                print(f'   - {k} checkpoint: {tests[k]:.4f} W/kg')
    print()


def gpu_line():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                              '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
        print('GPUs (index, busy, memory used, total):')
        for l in out.splitlines():
            print('  ' + l)
        print()
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('configs', nargs='*', default=DEFAULT)
    ap.add_argument('--last', type=int, default=10, help='validated epochs to show (default 10)')
    ap.add_argument('--plot', action='store_true', help='also draw live_training_curves.png')
    args = ap.parse_args()
    print(f'Run check at {dt.datetime.now():%Y-%m-%d %H:%M}\n')
    gpu_line()
    for c in args.configs:
        report(c, args.last)
    if args.plot:
        py = os.path.expanduser('~/miniconda3/envs/pyskl_310/bin/python')
        subprocess.run([py, os.path.join(ROOT, 'plot_live_curves.py')] + args.configs, cwd=ROOT,
                       env={**os.environ, 'PYTHONPATH': ROOT})


if __name__ == '__main__':
    sys.exit(main())
