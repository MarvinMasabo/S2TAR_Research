"""Build the prosthetic-only Watts/kg 'exact' pkl, same recipe as build_method_pkls.py
used for AbleBody2, but for the prosthetic (TAMUSA / PRO_A&M) cohort.

Source: final_demographics.pkl -- newest and most complete of the three candidate
prosthetic files on disk (see notes below); same annotation schema as AbleBody2
(gait='Prosthesis', has exact_windowing_energy_watts + weight + an existing
train/val/test split), just far fewer participants (3, not 22).
"""
import copy
import os
import pickle
import sys

import numpy as np

# final_demographics.pkl was pickled under numpy>=2.0 (references the numpy._core module
# path); this project's training env (pyskl_310) has numpy<2.0, which has no such module and
# fails with "No module named 'numpy._core'". Alias it to the equivalent numpy.core path
# before unpickling so this script can load the source directly in the training env -- the
# arrays that come back are then genuine old-numpy ndarrays, so the *output* pkl this script
# writes needs no special handling to be read back by pyskl/PoseDataset later.
sys.modules.setdefault('numpy._core', np.core)
sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
sys.modules.setdefault('numpy._core.umath', np.core.umath)
sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)

SRC = '/home/students/mmasabo1/summer26Research/final_demographics.pkl'
OUT_DIR = '/home/students/mmasabo1/summer26Research/prosthetic_wattkg'
FIELD = 'exact_windowing_energy_watts'   # the only method still in use per team decision


def finite(x):
    return x is not None and np.isfinite(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SRC, 'rb') as f:
        d = pickle.load(f)

    ann_all = d['annotations']
    base_split = {k: list(d['split'][k]) for k in ('train', 'val', 'test')}
    names = sorted(set(a['name'] for a in ann_all))
    print(f'source: {SRC}')
    print(f'{len(ann_all)} annotations, {len(names)} participants: {names}')
    print(f'splits={{ {", ".join(f"{k}:{len(v)}" for k, v in base_split.items())} }}')

    kept, dropped = [], []
    for a in ann_all:
        w, e = a.get('weight'), a.get(FIELD)
        if not (finite(w) and finite(e) and w > 0):
            dropped.append(a['frame_dir'])
            continue
        b = copy.deepcopy(a)
        b['label'] = float(e) / float(w)          # Watts / kg
        kept.append(b)

    keep_ids = {a['frame_dir'] for a in kept}
    new_split = {k: [fd for fd in v if fd in keep_ids] for k, v in base_split.items()}

    out = dict(split=new_split, annotations=kept)
    out.update({k: new_split[k] for k in ('train', 'val', 'test')})

    path = os.path.join(OUT_DIR, 'exact.pkl')
    with open(path, 'wb') as f:
        pickle.dump(out, f)

    labs = np.array([a['label'] for a in kept])
    print(f'kept={len(kept)} dropped={len(dropped)}  '
          f'splits={{ {", ".join(f"{k}:{len(v)}" for k, v in new_split.items())} }}  '
          f'label W/kg: min={labs.min():.3f} max={labs.max():.3f} mean={labs.mean():.3f}  '
          f'-> {path}')


if __name__ == '__main__':
    main()
