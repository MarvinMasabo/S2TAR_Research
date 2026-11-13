"""Demographics check: do the age / sex / height / weight values agree across each participant's
source sheet, and which clips in a dataset carry values that disagree?

    python report_demographics.py                      # checks combined_wattkg/exact_combined.pkl
    python report_demographics.py PKL [PKL ...]        # any of our pkls (able-bodied, prosthetic, combined)

For every participant, the "usual" value of each field is the one on most rows of their sheet (and on
most of their clips in the pkl). The report lists
  1. sheet rows whose value differs from the participant's usual value (file, Excel row, speed, time),
  2. the clips built from those rows (frame_dir, split, window),
  3. clips whose values appear nowhere in their sheet (should be 0: that would be a pipeline error).
Outputs (they contain participant names, so the folder is git-ignored):
  Gait Metabolic Results/Data Checks/Demographics Check.txt
  Gait Metabolic Results/Data Checks/demographic_discrepancies_sheet_rows.csv
  Gait Metabolic Results/Data Checks/demographic_discrepancies_clips.csv
"""
import glob
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

ROOT = Path('/home/students/mmasabo1/summer26Research')
SHEET_DIRS = {
    'able-bodied': ['/shared/research_rj/S2TAR Research/AbleBody2/Labels - Preprocessed/'
                    'AbleBody_csv_files_Final/AbleBody_csv_Labels_Final/*.csv'],
    'prosthetic': ['/shared/research_rj/S2TAR Research/Proshetic2/Labels/A&M/CSV/* Labels_windowed.csv',
                   '/shared/research_rj/S2TAR Research/Proshetic2/Labels/Thailand/CSV/* Labels_windowed.csv'],
}
DEFAULT_PKLS = [ROOT / 'combined_wattkg' / 'exact_combined.pkl']
OUT_DIR = ROOT / 'Gait Metabolic Results' / 'Data Checks'
FIELDS = ('age', 'gender', 'height', 'weight')
TOL = 1e-3   # sheets store 170.18, pkls 170.1799927


def read_sheet(path):
    try:
        df = pd.read_csv(path, encoding='utf-8-sig')
    except UnicodeDecodeError:            # a few able-bodied sheets were saved as Windows-1252
        df = pd.read_csv(path, encoding='cp1252')
    df.columns = [c.replace(' ', '') for c in df.columns]   # 'Weight (Kg)' and 'Weight(Kg)' both occur
    return df


def same(u, v):
    return abs(float(u) - float(v)) < TOL


def load_sheets():
    """(population, name) -> list of sheet rows with their demographics."""
    rows = defaultdict(list)
    n_files = Counter()
    for pop, patterns in SHEET_DIRS.items():
        for pattern in patterns:
            for path in sorted(glob.glob(pattern)):
                n_files[pop] += 1
                df = read_sheet(path)
                for i, r in df.iterrows():
                    if pd.isna(r.get('Name')) or pd.isna(r.get('Age')):
                        continue                                  # blank rows at the end of a sheet
                    est = r.get('EstimatedTime')
                    rows[(pop, str(r['Name']).strip())].append(dict(
                        file=Path(path).name, excel_row=i + 2, speed=str(r.get('Modality', '')).strip(),
                        estimated_time=None if pd.isna(est) else int(round(float(est))),
                        age=float(r['Age']), gender=1.0 if str(r['Sex']).strip().upper().startswith('M') else 0.0,
                        height=float(r['Height(cm)']), weight=float(r['Weight(Kg)'])))
    return rows, n_files


def usual(records):
    """The most common value of each field."""
    return {f: Counter(round(x[f], 3) for x in records).most_common(1)[0][0] for f in FIELDS}


def population_of(a):
    if 'population' in a:
        return a['population']
    return 'prosthetic' if 'source' in a else 'able-bodied'


def check_pkl(pkl_path, sheets, lines):
    d = pickle.load(open(pkl_path, 'rb'))
    ann = d['annotations']
    split_of = {fd: s for s in ('train', 'val', 'test') for fd in d['split'][s]}
    say = lines.append
    say(f'\n{"=" * 78}\nDATASET  {pkl_path}\n{"=" * 78}')
    say(f'{len(ann)} clips')

    # 1. completeness and coding
    say('\n[1] Missing values per field')
    for pop in sorted({population_of(a) for a in ann}):
        part = [a for a in ann if population_of(a) == pop]
        miss = {f: sum(a.get(f) is None or (isinstance(a.get(f), float) and math.isnan(a[f])) for a in part)
                for f in ('name', 'file_number', 'gait') + FIELDS}
        say(f'    {pop:<12} n={len(part):5d}  ' + '  '.join(f'{f}: {n}' for f, n in miss.items()))
        say(f"    {'':<12} sex coded as {sorted({a['gender'] for a in part})} (1 = male, 0 = female)")

    # group clips by participant
    clips = defaultdict(list)
    for a in ann:
        clips[(population_of(a), a['name'])].append(a)

    # 2. sheet rows that disagree with the participant's usual value
    sheet_rows = []
    for key in sorted(clips):
        recs = sheets.get(key, [])
        if not recs:
            continue
        exp = usual(recs)
        for x in recs:
            for f in FIELDS:
                if not same(x[f], exp[f]):
                    used = [a['frame_dir'] for a in clips[key]
                            if a['modality'] == x['speed'] and x['estimated_time'] is not None
                            and int(a['frame_dir'][-4:]) == x['estimated_time']]
                    sheet_rows.append(dict(population=key[0], participant=key[1], file=x['file'],
                                           excel_row=x['excel_row'], speed=x['speed'],
                                           estimated_time=x['estimated_time'], field=f,
                                           usual_value=exp[f], sheet_value=round(x[f], 3), clips_using_row=len(used)))
    say(f'\n[2] Sheet rows whose value differs from that participant\'s usual value: {len(sheet_rows)}')
    if sheet_rows:
        df = pd.DataFrame(sheet_rows)
        df['estimated_time'] = df['estimated_time'].astype('Int64')   # 300, not 300.0; blank stays <NA>
        say(df.drop(columns='population').to_string(index=False))

    # 3. clips carrying a value that differs from the participant's usual value
    clip_rows = []
    for key, part in sorted(clips.items()):
        exp = usual(part)
        for a in part:
            for f in FIELDS:
                if not same(a[f], exp[f]):
                    clip_rows.append(dict(population=key[0], participant=key[1], speed=a['modality'],
                                          frame_dir=a['frame_dir'], window_s=a['frame_dir'][-9:],
                                          split=split_of.get(a['frame_dir']), field=f,
                                          usual_value=exp[f], clip_value=round(a[f], 3)))
    affected = {r['frame_dir'] for r in clip_rows}
    say(f'\n[3] Clips whose value differs from that participant\'s usual value: {len(affected)} clips '
        f'({len({r["participant"] for r in clip_rows})} participants)')
    if clip_rows:
        say(pd.DataFrame(clip_rows).drop(columns='population').to_string(index=False))

    # 4. traceability: every clip's values should exist somewhere in its sheet
    untraced, no_sheet = [], set()
    for key, part in clips.items():
        recs = sheets.get(key)
        if not recs:
            no_sheet.add(key)
            continue
        for a in part:
            if not any(all(same(a[f], x[f]) for f in FIELDS) for x in recs):
                untraced.append(a['frame_dir'])
    say(f'\n[4] Clips whose values appear nowhere in their sheet: {len(untraced)}'
        + (f'  e.g. {untraced[:5]}' if untraced else ''))
    if no_sheet:
        say(f'    Participants with no sheet found: {sorted(no_sheet)}')

    weight_hit = any(r['field'] == 'weight' for r in clip_rows)
    say('\n[impact] ' + ('WEIGHT differs on some clips -> their W/kg label (energy / weight) needs checking.'
                         if weight_hit else
                         'No weight discrepancies, so every W/kg label (energy / weight) is unaffected. '
                         'STGCN / CTR-GCN train on skeletons only, so model results are unaffected too.'))
    for r in sheet_rows:
        r['dataset'] = Path(pkl_path).name
    for r in clip_rows:
        r['dataset'] = Path(pkl_path).name
    return sheet_rows, clip_rows


def main():
    pkls = [Path(p) for p in sys.argv[1:]] or DEFAULT_PKLS
    sheets, n_files = load_sheets()
    lines = ['Demographics check: source sheets vs dataset pkls',
             f"Sheets read: {n_files['able-bodied']} able-bodied, {n_files['prosthetic']} prosthetic "
             f"({sum(k[0] == 'able-bodied' for k in sheets)} + {sum(k[0] == 'prosthetic' for k in sheets)} participants)"]
    all_sheet, all_clips = [], []
    for p in pkls:
        s, c = check_pkl(p, sheets, lines)
        all_sheet += s
        all_clips += c

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = '\n'.join(lines)
    (OUT_DIR / 'Demographics Check.txt').write_text(report + '\n')
    sheet_df = pd.DataFrame(all_sheet)
    if len(sheet_df):
        sheet_df['estimated_time'] = sheet_df['estimated_time'].astype('Int64')
    sheet_df.to_csv(OUT_DIR / 'demographic_discrepancies_sheet_rows.csv', index=False)
    pd.DataFrame(all_clips).to_csv(OUT_DIR / 'demographic_discrepancies_clips.csv', index=False)
    print(report)
    print(f'\nSaved to {OUT_DIR}/')


if __name__ == '__main__':
    main()
