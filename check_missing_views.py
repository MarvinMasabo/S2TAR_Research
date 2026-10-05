"""List 10-s windows that were not captured by all three cameras (Back/Left/Right).

Reads Siem's original keypoint files, so a gap here comes from the source videos, not the merge.
    python check_missing_views.py
"""
import collections
import pickle
import re
import sys

import numpy as np

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

SIEM = '/shared/research_rj/Preprocessing_Code/Proshetic2/'
MERGED = '/home/students/mmasabo1/summer26Research/prosthetic_wattkg/exact_merged.pkl'
ANGLES = ('Back', 'Left', 'Right')


def trial_and_angle(fd):
    """'EE11_Fast_right_0290-0300' -> ('EE11_Fast', 'Right')."""
    m = re.match(r'(.+)_(back|left|right|unknown)_\d{4}-\d{4}$', fd, re.IGNORECASE)
    return m[1], m[2].title()


def main():
    merged = pickle.load(open(MERGED, 'rb'))
    trial_of = {a['frame_dir']: (a['source'], a['subject'], a['modality']) for a in merged['annotations']}
    split_of = {fd: s for s, v in merged['split'].items() for fd in v}

    views = collections.defaultdict(dict)          # (site, participant, speed, window) -> {angle: frame_dir}
    for site, f in [('TAMUSA', 'Full_Pros_TAMUSA.pkl'), ('Thailand', 'Full_Pros_Thailand.pkl')]:
        for a in pickle.load(open(SIEM + f, 'rb'))['annotations']:
            fd = a['frame_dir']
            if fd not in trial_of:
                continue                            # no label -> not in the dataset at all
            _, angle = trial_and_angle(fd)
            views[trial_of[fd] + (fd[-9:],)][angle] = (fd, a['total_frames'])

    last = {}
    for (site, who, speed, win) in views:
        last[(site, who, speed)] = max(last.get((site, who, speed), ''), win)

    rows = []
    for key, v in sorted(views.items()):
        if 'Unknown' in v:
            continue                                # EE03 was filmed by one camera only
        missing = [a for a in ANGLES if a not in v]
        if missing:
            site, who, speed, win = key
            any_fd = next(iter(v.values()))[0]
            rows.append((site, who, speed, win, win == last[key[:3]], sorted(v), missing, split_of[any_fd],
                         {a: v[a][1] for a in v}))

    print(f'{"site":8s} {"participant":15s} {"speed":9s} {"window":10s} {"last?":5s} '
          f'{"has":18s} {"missing":14s} {"split":5s}  frames per clip')
    for site, who, speed, win, is_last, has, missing, split, frames in rows:
        print(f'{site:8s} {who:15s} {speed:9s} {win:10s} {"yes" if is_last else "no":5s} '
              f'{",".join(has):18s} {",".join(missing):14s} {split:5s}  {frames}')
    print(f'\n{len(rows)} windows are missing a camera view; '
          f'{sum(r[4] for r in rows)} of them are the last window of their trial.')


if __name__ == '__main__':
    main()
