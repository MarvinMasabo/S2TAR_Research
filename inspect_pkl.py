"""Print a pkl's structure the same way the team's inspecting_pkl notebook does.

    python inspect_pkl.py PATH [PATH ...]
"""
import pickle
import sys
from collections import Counter

import numpy as np

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)


def inspect(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print('#' * 70)
    print(path)
    print('Type:', type(data))
    print('\nTop-level keys:')
    for key in data:
        print(' -', key)
    print('\nKey types:')
    for key, value in data.items():
        try:
            print(f'{key}: {type(value)} | length = {len(value)}')
        except TypeError:
            print(f'{key}: {type(value)}')

    ann = data['annotations']
    print('\n' + '=' * 60 + '\nFIRST ANNOTATION\n' + '=' * 60)
    for k, v in ann[0].items():
        print(f'  {k}: {type(v).__name__}, shape={v.shape}' if hasattr(v, 'shape') else f'  {k}: {repr(v)[:150]}')

    print('\n' + '=' * 60 + '\nFIELD TYPES ACROSS ALL ANNOTATIONS (missing = None/NaN)\n' + '=' * 60)
    for k in ann[0]:
        types = {type(a.get(k)).__name__ for a in ann}
        missing = sum(1 for a in ann if a.get(k) is None or (isinstance(a.get(k), float) and np.isnan(a.get(k))))
        print(f'{k:33s}: {types}' + (f'   missing {missing}/{len(ann)}' if missing else ''))
    print()


if __name__ == '__main__':
    for p in sys.argv[1:]:
        inspect(p)
