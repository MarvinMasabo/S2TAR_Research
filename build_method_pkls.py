"""Build per-method W/kg pkls from AbleBody2_full.pkl.

For each method the regression target `label` is set to
    <method energy field, Watts> / <subject weight, kg>   ->  Watts/kg

Rows with a missing / non-finite energy value or weight are dropped, and the
split id lists are filtered to match. All other annotation fields are kept.
"""
import copy
import os
import pickle

import numpy as np

SRC = '/home/students/mmasabo1/summer26Research/AbleBody2_full.pkl'
OUT_DIR = '/home/students/mmasabo1/summer26Research/ablebody2_wattkg'

METHODS = {
    'nearest':   'nearest_ee_watts',
    'exact':     'exact_windowing_energy_watts',
    'windowing': 'windowing_drop_avg_energy_watts',
}


def finite(x):
    return x is not None and np.isfinite(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SRC, 'rb') as f:
        d = pickle.load(f)

    ann_all = d['annotations']
    base_split = {k: list(d['split'][k]) for k in ('train', 'val', 'test')}
    print(f'source: {len(ann_all)} annotations  '
          f'splits={{ {", ".join(f"{k}:{len(v)}" for k, v in base_split.items())} }}')

    for method, field in METHODS.items():
        kept, dropped = [], []
        for a in ann_all:
            w, e = a.get('weight'), a.get(field)
            if not (finite(w) and finite(e) and w > 0):
                dropped.append(a['frame_dir'])
                continue
            b = copy.deepcopy(a)
            b['label'] = float(e) / float(w)          # Watts / kg
            kept.append(b)

        keep_ids = {a['frame_dir'] for a in kept}
        new_split = {k: [fd for fd in v if fd in keep_ids]
                     for k, v in base_split.items()}

        out = dict(split=new_split, annotations=kept)
        out.update({k: new_split[k] for k in ('train', 'val', 'test')})  # mirror top-level lists

        path = os.path.join(OUT_DIR, f'{method}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(out, f)

        labs = np.array([a['label'] for a in kept])
        print(f'[{method:9s}] field={field:32s} kept={len(kept)} dropped={len(dropped)}  '
              f'splits={{ {", ".join(f"{k}:{len(v)}" for k, v in new_split.items())} }}  '
              f'label W/kg: min={labs.min():.3f} max={labs.max():.3f} mean={labs.mean():.3f}  '
              f'-> {path}')


if __name__ == '__main__':
    main()
