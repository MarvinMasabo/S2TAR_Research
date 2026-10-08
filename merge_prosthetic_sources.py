import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# numpy>=2.0 compat shim -- see build_prosthetic_pkl.py for why this is needed.
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.umath', np.core.umath)
    sys.modules.setdefault('numpy._core._multiarray_umath', np.core._multiarray_umath)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path('/home/students/mmasabo1/summer26Research')
SIEM_DIR = Path('/shared/research_rj/Preprocessing_Code/Proshetic2')
KEYPOINT_PKLS = {'TAMUSA': SIEM_DIR / 'Full_Pros_TAMUSA.pkl', 'Thailand': SIEM_DIR / 'Full_Pros_Thailand.pkl'}
LABEL_DIRS = {'TAMUSA': SIEM_DIR / 'Labels/A&M/CSV', 'Thailand': SIEM_DIR / 'Labels/Thailand/CSV'}
TAMUSA_STEMS = ROOT / 'final_demographics.pkl'   # TAMUSA video code -> speed (codes like '1_2' aren't speeds)
OUTPUT_PATH = ROOT / 'prosthetic_wattkg' / 'exact_merged.pkl'

WINDOW = 10
VAL_RATIO = TEST_RATIO = 0.1
EXACT = 'Exact Windowing Energy (Watts)'
THAI_MODALITY = {'Slow': 'Slower', 'Normal': 'Preferred', 'Fast': 'Faster'}


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def angle_of(frame_dir):
    m = re.search(r'_(back|left|right)_', frame_dir, re.IGNORECASE)
    return m.group(1).title() if m else 'Unknown'


def build_stem_map():
    """clip stem (frame_dir minus window) -> (site, participant, speed)."""
    stem_map = {}
    for a in load(TAMUSA_STEMS)['annotations']:
        stem = a['frame_dir'][:-10]
        stem_map[stem] = ('TAMUSA', a['name'], a['modality'])
    n_tam = len(stem_map)
    for a in load(KEYPOINT_PKLS['Thailand'])['annotations']:
        m = re.match(r'(EE\d+)_(Slow|Normal|Fast)_', a['frame_dir'])
        stem_map[a['frame_dir'][:-10]] = ('Thailand', m[1], THAI_MODALITY[m[2]])
    print(f'[stems]   Loaded {len(stem_map)} stem→(participant, speed) entries '
          f'(TAMUSA {n_tam}, Thailand {len(stem_map) - n_tam})')
    return stem_map


# windowed-CSV column -> field name, the same names AbleBody2_full.pkl and final_demographics.pkl use
ENERGY_COLUMNS = {
    'Watts': 'watts',
    'Nearest EE (Watts)': 'nearest_ee_watts',
    'Exact Windowing Energy (Watts)': 'exact_windowing_energy_watts',
    'Windowing Drop Avg Energy (Watts)': 'windowing_drop_avg_energy_watts',
    'Nearest HR (bpm)': 'heart_rate',
    'Exact Windowing HR (bpm)': 'exact_windowing_hr',
    'Windowing Drop Avg HR (bpm)': 'windowing_drop_avg_hr',
    'VO2/kg  STPD (ml/kg/m)': 'vo2_per_kg',
    'VO2  STPD (L/min)': 'vo2_stpd',
    'VCO2 STPD (L/min)': 'vco2_stpd',
    'RER': 'rer',
    'METS': 'mets',
    'Energy (cal)': 'energy_cal',      # empty / absent in the prosthetic files -> NaN
    'VE STPD (L/min)': 've_stpd',      # prosthetic files only have VE BTPS (A&M) -> NaN, not relabelled
}
# AbleBody2_full.pkl's per-clip fields, in its order; our bookkeeping fields follow them
FIELD_ORDER = ['frame_dir', 'label', 'img_shape', 'total_frames', 'num_person_raw', 'keypoint', 'keypoint_score',
               'name', 'file_number', 'gait', 'modality', 'age', 'gender', 'height', 'weight', 'watts', 'energy_cal',
               'nearest_ee_watts', 'exact_windowing_energy_watts', 'windowing_drop_avg_energy_watts', 'heart_rate',
               'exact_windowing_hr', 'windowing_drop_avg_hr', 'vo2_per_kg', 'rer', 've_stpd', 'mets', 'vo2_stpd',
               'vco2_stpd']
EXTRA_FIELDS = ['source', 'subject', 'angle', 'moment']
WINDOWING = ('Exact Windowing Energy (Watts)', 'Nearest EE (Watts)', 'Windowing Drop Avg Energy (Watts)')


def num(v):
    return float(v) if pd.notna(v) else np.nan


def build_label_lookup():
    """(site, participant, speed, clip_start_sec) -> every windowing method + demographics, from the windowed CSVs.

    Rows with any windowing value are kept; the label (Exact Windowing / weight) is NaN where the
    Exact Windowing value is missing, and those clips are skipped later with that reason."""
    lookup, n_files, n_rows = {}, 0, 0
    for site, d in LABEL_DIRS.items():
        for path in sorted(d.glob('* Labels_windowed.csv')):
            df = pd.read_csv(path, encoding='utf-8-sig')
            n_files += 1
            n_rows += len(df)
            df = df[df['Estimated Time'].notna() & (df['Estimated Time'] > 0) & df[list(WINDOWING)].notna().any(axis=1)]
            for _, r in df.iterrows():
                who = r['Name'] if site == 'TAMUSA' else f"EE{int(r['File Number']):02d}"
                start = int(round(r['Estimated Time'])) - WINDOW
                w = float(r['Weight(Kg)'])
                entry = {
                    'label': num(r[EXACT]) / w,
                    'name': str(r['Name']).strip(),
                    'file_number': int(r['File Number']),
                    'gait': str(r['Gait']).strip(),
                    'age': float(r['Age']),
                    'gender': 1.0 if str(r['Sex']).strip().upper() == 'M' else 0.0,
                    'height': float(r['Height (cm)']),
                    'weight': w,
                }
                entry.update({field: num(r.get(col)) for col, field in ENERGY_COLUMNS.items()})  # absent column -> NaN
                lookup[(site, who, str(r['Modality']).strip(), start)] = entry
    n_exact = sum(np.isfinite(v['label']) for v in lookup.values())
    print(f'[labels]  Loaded {n_files} CSV files → {n_rows} rows')
    print(f'[labels]  Label lookup: {len(lookup)} windows with a windowing value '
          f'({n_exact} with Exact Windowing, used as the label)')
    return lookup


def main_person(a):
    kp, sc = a['keypoint'], a['keypoint_score']
    i = int(np.argmax(sc.reshape(len(sc), -1).mean(1)))
    return kp[i:i + 1].astype(np.float32), sc[i:i + 1].astype(np.float32)


def build_annotations(stem_map, lookup):
    annotations, skipped = [], []
    for site, path in KEYPOINT_PKLS.items():
        clips = load(path)['annotations']
        print(f'[keypts]  Loaded {len(clips)} clips from {path.name}')
        for a in clips:
            fd = a['frame_dir']
            stem, start = fd[:-10], int(fd[-9:-5])
            if stem not in stem_map:
                skipped.append((fd, f'{site}: video not in speed map'))
                continue
            _, who, modality = stem_map[stem]
            key = (site, who, modality, start)
            if key not in lookup:
                reason = ('no label CSV for this participant'
                          if not any(k[:2] == (site, who) for k in lookup)
                          else 'no windowing value for this window')
                skipped.append((fd, f'{site} {who} {modality}: {reason}'))
                continue
            if not np.isfinite(lookup[key]['label']):
                skipped.append((fd, f'{site} {who} {modality}: Nearest/Drop-Avg value only, no Exact Windowing value'))
                continue
            kp, sc = main_person(a)
            annotations.append(dict(
                frame_dir=fd, img_shape=tuple(a['img_shape']),
                total_frames=int(a['total_frames']), num_person_raw=int(a['keypoint'].shape[0]),
                keypoint=kp, keypoint_score=sc, **lookup[key],
                source=site, subject=who, modality=modality, angle=angle_of(fd),
                moment=f'{site}|{who}|{modality}|{start:04d}', start_sec=start))
    print(f'[merge]   Built: {len(annotations)}  |  Skipped: {len(skipped)}')
    for reason, n in sorted(Counter(r for _, r in skipped).items()):
        print(f'            SKIP {n:4d}  {reason}')
    return annotations


def stratified_split(annotations):
    """Time-ordered 80/10/10 within each (site, participant, speed), by 10-s window."""
    windows = defaultdict(set)
    for a in annotations:
        windows[(a['source'], a['subject'], a['modality'])].add(a['start_sec'])
    split_of = {}
    for g, starts in sorted(windows.items()):
        starts = sorted(starts)
        n = len(starts)
        # round half up (not floor): identical for able-bodied 20-window trials, but floor(2.9)=2
        # would leave the 29-window prosthetic trials at ~7% val/test
        n_val = max(1, math.floor(n * VAL_RATIO + 0.5))
        n_test = max(1, math.floor(n * TEST_RATIO + 0.5))
        n_train = n - n_val - n_test
        for i, s in enumerate(starts):
            split_of[g + (s,)] = 'train' if i < n_train else 'val' if i < n_train + n_val else 'test'
    split = {s: [] for s in ('train', 'val', 'test')}
    for a in annotations:
        split[split_of[(a['source'], a['subject'], a['modality'], a['start_sec'])]].append(a['frame_dir'])

    sp = {fd: s for s, v in split.items() for fd in v}
    total = len(annotations)
    print(f"\n[split]   train={len(split['train'])} ({100 * len(split['train']) / total:.0f}%)  "
          f"val={len(split['val'])} ({100 * len(split['val']) / total:.0f}%)  "
          f"test={len(split['test'])} ({100 * len(split['test']) / total:.0f}%)  total={total}")
    for axis, key in [('Site', 'source'), ('Speed', 'modality'), ('Angle', 'angle'), ('Participant', 'subject')]:
        print(f'\n[split]   By {axis}:')
        print(f"  {'Value':<22}  {'Train':>6}  {'Val':>5}  {'Test':>5}  {'Total':>6}")
        print(f"  {'-' * 22}  {'-' * 6}  {'-' * 5}  {'-' * 5}  {'-' * 6}")
        for v in sorted({a[key] for a in annotations}):
            c = Counter(sp[a['frame_dir']] for a in annotations if a[key] == v)
            print(f"  {str(v):<22}  {c['train']:>6}  {c['val']:>5}  {c['test']:>5}  {sum(c.values()):>6}")
    return split


def save_pkl(split, annotations):
    annotations = [{k: a[k] for k in FIELD_ORDER + EXTRA_FIELDS} for a in annotations]  # AbleBody2 order
    out = dict(split=split, annotations=annotations, **split)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(out, f)
    print(f'\n[output]  Saved → {OUTPUT_PATH}')
    print(f'          annotations   : {len(annotations)}')
    print(f"          train/val/test: {len(split['train'])} / {len(split['val'])} / {len(split['test'])}")


def verify_pkl():
    pkl = load(OUTPUT_PATH)
    print(f'\n[verify]  Keys: {list(pkl.keys())}')
    for s in ('train', 'val', 'test'):
        print(f"[verify]    {s}: {len(pkl['split'][s])} frame_dirs")
    sp = {fd: s for s, v in pkl['split'].items() for fd in v}
    moments = defaultdict(set)
    for a in pkl['annotations']:
        moments[a['moment']].add(sp[a['frame_dir']])
    print(f'[verify]  Moments (10-s windows) with camera views in different splits: '
          f'{sum(len(v) > 1 for v in moments.values())} of {len(moments)}')
    for site in ('TAMUSA', 'Thailand'):
        sample = next(a for a in pkl['annotations'] if a['source'] == site)
        print(f'\n[verify]  Sample annotation ({site}):')
        print(f"          frame_dir      : {sample['frame_dir']}")
        print(f"          label (W/kg)   : {sample['label']:.4f}")
        print(f"          keypoint shape : {np.array(sample['keypoint']).shape}")
        print(f"          age / gender   : {sample['age']} / {sample['gender']}")
        print(f"          height / weight: {sample['height']} / {sample['weight']}")
        print(f"          exact / nearest / drop-avg (W): {sample['exact_windowing_energy_watts']} / "
              f"{sample['nearest_ee_watts']} / {sample['windowing_drop_avg_energy_watts']}")
        print(f"          HR nearest / exact (bpm)   : {sample['heart_rate']} / {sample['exact_windowing_hr']}")
    fields = [k for k in pkl['annotations'][0] if k not in ('keypoint', 'keypoint_score')]
    print(f'\n[verify]  Fields per clip ({len(fields)}): {fields}')


def main():
    print('=' * 60)
    print('  Prosthetic (TAMUSA + Thailand) — building ONE pickle')
    print('=' * 60)
    stem_map = build_stem_map()
    lookup = build_label_lookup()
    annotations = build_annotations(stem_map, lookup)
    if not annotations:
        print('\n[ERROR] No annotations built. Check paths above.')
        return
    split = stratified_split(annotations)
    save_pkl(split, annotations)
    verify_pkl()
    print('\nDone.')


if __name__ == '__main__':
    main()
