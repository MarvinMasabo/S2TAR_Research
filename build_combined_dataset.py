"""Build ONE combined pickle (able-bodied + prosthetic) for training a single model on both
populations.

  inputs  = ablebody2_wattkg/exact.pkl            (22 able-bodied participants)
            prosthetic_wattkg/exact_merged.pkl    (14 prosthesis users, TAMUSA + Thailand)
  split   = each population keeps the split it already has (both are time-ordered 80/10/10 with
            every camera view of a 10-s window in the same split); the lists are concatenated, so
            the combined test set is exactly able-bodied test + prosthetic test.
  label   = unchanged: Exact Windowing energy / body weight (W/kg) in both inputs.
  fields  = one shared set for every clip (training fields + population, source, subject, speed,
            angle, moment, demographics, all three windowing methods with their heart rates, and the
            gas measurements), named as in AbleBody2_full.pkl.
"""
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

ROOT = Path('/home/students/mmasabo1/summer26Research')
ABLE_PKL = ROOT / 'ablebody2_wattkg' / 'exact.pkl'
PROS_PKL = ROOT / 'prosthetic_wattkg' / 'exact_merged.pkl'
ABLE_TRIM_SHEETS = [Path('/shared/research_rj/Preprocessing_Code/AbleBody2/util/labels.csv'),
                    Path('/shared/research_rj/Preprocessing_Code/AbleBody2/util/labels-2.csv')]
OUTPUT_PATH = ROOT / 'combined_wattkg' / 'exact_combined.pkl'

SPLITS = ('train', 'val', 'test')
TRAIN_FIELDS = ('frame_dir', 'label', 'img_shape', 'total_frames', 'num_person_raw', 'keypoint', 'keypoint_score')
# demographics + every windowing method + gas measurements: the same field names in both input files
SHARED_FIELDS = ('name', 'file_number', 'gait', 'age', 'gender', 'height', 'weight',
                 'watts', 'nearest_ee_watts', 'exact_windowing_energy_watts', 'windowing_drop_avg_energy_watts',
                 'heart_rate', 'exact_windowing_hr', 'windowing_drop_avg_hr',
                 'vo2_per_kg', 'vo2_stpd', 'vco2_stpd', 'rer', 'mets', 'energy_cal', 've_stpd')
# AbleBody2_full.pkl's per-clip fields in its order, then the bookkeeping fields
FIELD_ORDER = ['frame_dir', 'label', 'img_shape', 'total_frames', 'num_person_raw', 'keypoint', 'keypoint_score',
               'name', 'file_number', 'gait', 'modality', 'age', 'gender', 'height', 'weight', 'watts', 'energy_cal',
               'nearest_ee_watts', 'exact_windowing_energy_watts', 'windowing_drop_avg_energy_watts', 'heart_rate',
               'exact_windowing_hr', 'windowing_drop_avg_hr', 'vo2_per_kg', 'rer', 've_stpd', 'mets', 'vo2_stpd',
               'vco2_stpd', 'population', 'source', 'subject', 'angle', 'moment']


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def able_angle_map():
    """(folder, video stem) -> camera angle, from the trim sheets that cut the able-bodied videos."""
    angles = {}
    for sheet in ABLE_TRIM_SHEETS:
        df = pd.read_csv(sheet, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        for folder, video, angle in zip(df['Folder'], df['Video'], df['angle']):
            stem = str(video).split(' - ')[0].strip().replace(' ', '_')
            angles[(str(folder).strip(), stem)] = str(angle).strip().title()
    return angles


def angle_from_name(stem):
    """Fallback when a video is missing from the trim sheets: Left/Right/Back anywhere in the name
    ('Ava_Bac_1_9' -> Back, 'CG008RIght_Slow' -> Right)."""
    s = stem.lower()
    for key, angle in (('left', 'Left'), ('right', 'Right'), ('back', 'Back'), ('_bac_', 'Back')):
        if key in s:
            return angle
    return 'Unknown'


def able_clips():
    d = load(ABLE_PKL)
    angles = able_angle_map()
    out, sources = [], Counter()
    for a in d['annotations']:
        folder, clip = a['frame_dir'].split('/', 1)
        stem, start = clip[:-10], int(clip[-9:-5])
        if (folder, stem) in angles:
            angle = angles[(folder, stem)]
            sources['trim sheet'] += 1
        else:
            angle = angle_from_name(stem)
            sources['clip name'] += 1
        c = {f: a[f] for f in TRAIN_FIELDS}
        c.update(population='able-bodied', source='AbleBody2', subject=a['name'], modality=a['modality'],
                 angle=angle, moment=f"AbleBody2|{a['name']}|{a['modality']}|{start:04d}",
                 **{f: a.get(f) for f in SHARED_FIELDS})
        out.append(c)
    print(f'[able]    {len(out)} clips from {ABLE_PKL.name}; camera angle from {dict(sources)}')
    return out, d['split']


def pros_clips():
    d = load(PROS_PKL)
    out = []
    for a in d['annotations']:
        c = {f: a[f] for f in TRAIN_FIELDS}
        c.update(population='prosthetic', source=a['source'], subject=a['subject'], modality=a['modality'],
                 angle=a['angle'], moment=a['moment'], **{f: a.get(f) for f in SHARED_FIELDS})
        out.append(c)
    print(f'[pros]    {len(out)} clips from {PROS_PKL.name}')
    return out, d['split']


def table(title, ann, split_of, key):
    print(f'\n[split]   By {title}:')
    print(f"  {'Value':<24}  {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Total':>6}   {'train/val/test %':>18}")
    for v in sorted({key(a) for a in ann}, key=str):
        c = Counter(split_of[a['frame_dir']] for a in ann if key(a) == v)
        n = sum(c.values())
        pct = '/'.join(f'{100 * c[s] / n:.0f}' for s in SPLITS)
        print(f"  {str(v):<24}  {c['train']:>6}  {c['val']:>5}  {c['test']:>5}  {n:>6}   {pct:>18}")


def verify(ann, split):
    names = [a['frame_dir'] for a in ann]
    listed = [fd for s in SPLITS for fd in split[s]]
    print('\n[verify]  checks')
    print(f'  clip names unique ................ {len(set(names)) == len(names)}')
    print(f'  every clip in exactly one split .. {sorted(listed) == sorted(names)}')
    split_of = {fd: s for s in SPLITS for fd in split[s]}
    moments = defaultdict(set)
    for a in ann:
        moments[a['moment']].add(split_of[a['frame_dir']])
    print(f"  moments split across sets ........ {sum(len(v) > 1 for v in moments.values())} of {len(moments)}")
    print(f"  keypoints all float32, 1 person .. "
          f"{all(a['keypoint'].dtype == np.float32 and a['keypoint'].shape[0] == 1 for a in ann)}")
    print(f"  labels all finite ................ {all(np.isfinite(a['label']) for a in ann)}")
    print('\n[labels]  W/kg by population and split (mean / sd / min-max)')
    for pop in ('able-bodied', 'prosthetic'):
        for s in SPLITS:
            v = np.array([a['label'] for a in ann if a['population'] == pop and split_of[a['frame_dir']] == s])
            print(f'  {pop:<12} {s:<5} n={len(v):5d}  {v.mean():.2f} / {v.std():.2f} / {v.min():.2f}-{v.max():.2f}')
    return split_of


def main():
    print('=' * 60)
    print('  Able-bodied + prosthetic — building ONE combined pickle')
    print('=' * 60)
    able, able_split = able_clips()
    pros, pros_split = pros_clips()
    ann = able + pros
    split = {s: list(able_split[s]) + list(pros_split[s]) for s in SPLITS}
    total = len(ann)
    print(f"\n[merge]   {total} clips: train={len(split['train'])} ({100 * len(split['train']) / total:.0f}%)  "
          f"val={len(split['val'])} ({100 * len(split['val']) / total:.0f}%)  "
          f"test={len(split['test'])} ({100 * len(split['test']) / total:.0f}%)")

    split_of = verify(ann, split)
    table('Population', ann, split_of, lambda a: a['population'])
    table('Site', ann, split_of, lambda a: a['source'])
    table('Speed', ann, split_of, lambda a: a['modality'])
    table('Angle', ann, split_of, lambda a: a['angle'])
    table('Population x speed', ann, split_of, lambda a: f"{a['population']} {a['modality']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(dict(split=split, annotations=[{k: a[k] for k in FIELD_ORDER} for a in ann], **split), f)
    print(f'\n[output]  Saved -> {OUTPUT_PATH}')
    print(f'          {len({a["subject"] for a in ann})} participants, fields: '
          f'{[k for k in ann[0] if k not in ("keypoint", "keypoint_score")]}')
    print('\nDone.')


if __name__ == '__main__':
    main()
