"""Verify a prosthetic-cohort pkl's train/val/test split matches AbleBody2's approach:
80/10/10, stratified by participant, walking-speed modality, and camera angle, with every
camera view of the same moment in the same split.

Prints a report; exits non-zero if any stratum drifts too far from 80/10/10 (>15 points) or any
moment's views straddle splits, which would mean the split needs to be rebuilt.
"""
import argparse
import collections
import pickle
import re
import sys

TOLERANCE_PCT = 15.0   # how far a stratum's train% may drift from 80% before flagging it


def angle_of(frame_dir):
    m = re.search(r'_(Back|Left|Right|Front|Side)_', frame_dir, re.IGNORECASE)
    return m.group(1).title() if m else 'UNKNOWN'


def moment_of(frame_dir):
    """Fallback moment id for pkls without a 'moment' field: the clip name minus its camera angle."""
    return re.sub(r'_(Back|Left|Right|Front|Side|Unknown)_', '_', frame_dir, flags=re.IGNORECASE)


def stratum_report(name, groups, split_of):
    """groups: {stratum_value: [frame_dir, ...]}. Prints per-stratum split % and flags drift."""
    print(f'\n--- {name} ---')
    worst = 0.0
    for value, fds in sorted(groups.items()):
        counts = collections.Counter(split_of.get(fd, 'MISSING') for fd in fds)
        tot = len(fds)
        pct = {s: 100 * counts.get(s, 0) / tot for s in ('train', 'val', 'test')}
        drift = abs(pct['train'] - 80)
        worst = max(worst, drift)
        flag = '  <-- DRIFT' if drift > TOLERANCE_PCT else ''
        print(f'  {value:12s} n={tot:4d}  train={pct["train"]:5.1f}%  '
              f'val={pct["val"]:5.1f}%  test={pct["test"]:5.1f}%{flag}')
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl', help='path to the pkl to verify')
    args = ap.parse_args()

    with open(args.pkl, 'rb') as f:
        d = pickle.load(f)
    ann = {a['frame_dir']: a for a in d['annotations']}
    split = d['split']
    split_of = {fd: s for s in ('train', 'val', 'test') for fd in split[s]}

    print(f'Verifying: {args.pkl}')
    print(f'{len(ann)} clips, {len(split["train"])}/{len(split["val"])}/{len(split["test"])} '
          f'train/val/test ({100*len(split["train"])/len(ann):.1f}% / '
          f'{100*len(split["val"])/len(ann):.1f}% / {100*len(split["test"])/len(ann):.1f}%)')

    by_participant = collections.defaultdict(list)
    by_modality = collections.defaultdict(list)
    by_angle = collections.defaultdict(list)
    by_source = collections.defaultdict(list)
    for fd, a in ann.items():
        # CM's files carry no name/modality fields; fall back to the frame_dir's subject prefix
        by_participant[a.get('name') or a.get('subject') or fd.split('_')[0]].append(fd)
        by_modality[a.get('modality', 'UNKNOWN')].append(fd)
        by_angle[angle_of(fd)].append(fd)
        if 'source' in a:
            by_source[a['source']].append(fd)

    worst = 0.0
    if len(by_source) > 1:
        worst = max(worst, stratum_report('By source', by_source, split_of))
    worst = max(worst, stratum_report('By participant', by_participant, split_of))
    worst = max(worst, stratum_report('By modality', by_modality, split_of))
    worst = max(worst, stratum_report('By angle', by_angle, split_of))

    # Camera views of the same moment share a label; if they straddle splits, test leaks into train.
    moments = collections.defaultdict(set)
    for fd, a in ann.items():
        moments[a.get('moment') or moment_of(fd)].add(split_of.get(fd, 'MISSING'))
    leaked = sum(1 for s in moments.values() if len(s) > 1)
    print(f'\n--- Camera views of the same moment ---\n  {len(moments)} moments, '
          f'{leaked} with views in different splits')

    print()
    if worst <= TOLERANCE_PCT and leaked == 0:
        print(f'VERDICT: OK - stratified 80/10/10 within {TOLERANCE_PCT:.0f} points on every '
              f'participant/modality/angle (worst drift {worst:.1f} points), and no moment\'s '
              f'camera views are split across train/val/test. Ready to use as-is.')
        return 0
    problems = []
    if worst > TOLERANCE_PCT:
        problems.append(f'worst drift {worst:.1f} points exceeds {TOLERANCE_PCT:.0f}-point tolerance')
    if leaked:
        problems.append(f'{leaked} moments have camera views in different splits')
    print(f'VERDICT: NEEDS REBUILD - {"; ".join(problems)}.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
