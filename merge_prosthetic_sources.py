"""Merge prosthetic-cohort sources (TAMUSA/PRO_A&M, PRO_THAI, ...) into one Watts/kg
'exact windowing' pkl, the same way build_method_pkls.py / build_prosthetic_pkl.py did for
AbleBody2/TAMUSA individually.

Per Ricardo (2026-09 meeting): each source is split individually first (already verified
80/10/10, stratified by participant/modality/angle -- see verify_prosthetic_split.py), then
the splits are unioned rather than re-derived, so no source's stratification is disturbed.

Only TAMUSA/PRO_A&M (final_demographics.pkl) exists in this project as of this run --
PRO_THAI has not been added yet. Add its path to SOURCES the moment it exists; nothing else
in this script needs to change.
"""
import copy
import os
import pickle
import sys

import numpy as np

# numpy>=2.0 compat shim -- see build_prosthetic_pkl.py for why this is needed.
sys.modules.setdefault('numpy._core', np.core)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core.umath', np.core.umath)
sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)

ROOT = '/home/students/mmasabo1/summer26Research'
FIELD = 'exact_windowing_energy_watts'
OUT_PATH = os.path.join(ROOT, 'prosthetic_wattkg', 'exact_merged.pkl')

# label -> path. Add PRO_THAI here once it exists in the project.
SOURCES = {
    'PRO_A&M (TAMUSA)': os.path.join(ROOT, 'final_demographics.pkl'),
    'PRO_THAI': os.path.join(ROOT, 'PRO_THAI.pkl'),   # not present yet -- skipped if missing
}


def finite(x):
    return x is not None and np.isfinite(x)


def main():
    all_ann, all_split = [], {'train': [], 'val': [], 'test': []}
    used, skipped = [], []

    for label, path in SOURCES.items():
        if not os.path.exists(path):
            skipped.append((label, path))
            continue
        with open(path, 'rb') as f:
            d = pickle.load(f)
        kept = 0
        for a in d['annotations']:
            w, e = a.get('weight'), a.get(FIELD)
            if not (finite(w) and finite(e) and w > 0):
                continue
            b = copy.deepcopy(a)
            b['label'] = float(e) / float(w)
            b['source'] = label   # keep provenance visible in the merged file
            all_ann.append(b)
            kept += 1
        keep_ids = {a['frame_dir'] for a in d['annotations']
                    if finite(a.get('weight')) and finite(a.get(FIELD)) and a.get('weight') > 0}
        for s in ('train', 'val', 'test'):
            all_split[s].extend(fd for fd in d['split'][s] if fd in keep_ids)
        used.append((label, path, kept, {s: len(d['split'][s]) for s in ('train', 'val', 'test')}))

    print('=== sources included ===')
    for label, path, kept, splitsizes in used:
        print(f'  {label}: {kept} clips from {path}  (source split {splitsizes})')
    print('=== sources skipped (file not found) ===')
    for label, path in skipped:
        print(f'  {label}: expected at {path} -- NOT FOUND, excluded from merge')

    if not used:
        print('\nNo sources found at all -- nothing to merge.')
        return

    out = dict(split=all_split, annotations=all_ann)
    out.update(all_split)
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(out, f)

    tot = sum(len(v) for v in all_split.values())
    print(f'\nMerged: {len(all_ann)} clips, '
          f'{len(all_split["train"])}/{len(all_split["val"])}/{len(all_split["test"])} '
          f'train/val/test ({100*len(all_split["train"])/tot:.1f}%/'
          f'{100*len(all_split["val"])/tot:.1f}%/{100*len(all_split["test"])/tot:.1f}%)')
    print(f'-> {OUT_PATH}')
    if skipped:
        print(f'\nNOTE: only {len(used)}/{len(SOURCES)} configured source(s) were available. '
              f'Re-run this script once the missing source(s) are added -- everything else '
              f'(configs, training scripts) reads from {OUT_PATH}, so nothing downstream needs '
              f'to change when the merge is redone.')


if __name__ == '__main__':
    main()
