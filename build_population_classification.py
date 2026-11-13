"""Build the able-bodied vs prosthetic CLASSIFICATION pickle from the combined dataset.

  input   = combined_wattkg/exact_combined.pkl   (9,113 clips, built by build_combined_dataset.py)
  label   = 0 able-bodied, 1 prosthetic          (from the 'population' field)
  wattkg  = the regression label (Exact Windowing W/kg), kept for reference
  split   = unchanged: the combined split (each population's time-ordered 80/10/10)

Note: the split is by time within each participant, so every test participant also has earlier
windows in train. That makes this the "simple" version of the task; holding whole participants out
would be the stricter test.
"""
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

ROOT = Path('/home/students/mmasabo1/summer26Research')
COMBINED_PKL = ROOT / 'combined_wattkg' / 'exact_combined.pkl'
OUTPUT_PATH = ROOT / 'population_cls' / 'able_vs_prosthetic.pkl'
CLASSES = ('able-bodied', 'prosthetic')   # label = index


def main():
    with open(COMBINED_PKL, 'rb') as f:
        d = pickle.load(f)
    ann = []
    for a in d['annotations']:
        a = dict(a)
        a['wattkg'] = a['label']
        a['label'] = CLASSES.index(a['population'])
        ann.append(a)
    split = d['split']
    split_of = {fd: s for s in ('train', 'val', 'test') for fd in split[s]}

    print(f'[cls]     {len(ann)} clips from {COMBINED_PKL.name}; label 0 = {CLASSES[0]}, 1 = {CLASSES[1]}')
    print(f"  {'Split':<6} {'able-bodied':>12} {'prosthetic':>11} {'total':>6}")
    for s in ('train', 'val', 'test'):
        c = Counter(a['label'] for a in ann if split_of[a['frame_dir']] == s)
        print(f'  {s:<6} {c[0]:>12} {c[1]:>11} {c[0] + c[1]:>6}')
    subjects = {s: {a['subject'] for a in ann if split_of[a['frame_dir']] == s} for s in ('train', 'test')}
    print(f"[cls]     test participants also in train: {len(subjects['test'] & subjects['train'])} "
          f"of {len(subjects['test'])} (time-ordered split)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dict(split=split, annotations=ann, **split), f)
    print(f'[output]  Saved -> {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
