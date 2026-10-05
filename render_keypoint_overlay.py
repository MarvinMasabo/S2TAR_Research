"""Side-by-side video: HRNet skeleton drawn on clip A, next to clip B shown as-is (or with its
skeleton too, if it has keypoints). Also saves a still contact sheet.

    python render_keypoint_overlay.py EE04_Normal_back_0250-0260 EE04_Normal_back_0260-0270
"""
import glob
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np

# Siem's pkls were saved under numpy>=2.0 -- see build_prosthetic_pkl.py
if not hasattr(np, '_core'):
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)

CLIPS = '/shared/research_rj/clipsV1Trimmed'
PKLS = ['/shared/research_rj/Preprocessing_Code/Proshetic2/Full_Pros_TAMUSA.pkl',
        '/shared/research_rj/Preprocessing_Code/Proshetic2/Full_Pros_Thailand.pkl']
FFMPEG = os.path.expanduser('~/miniconda3/bin/ffmpeg')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# COCO-17 skeleton (HRNet output order)
EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
         (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]
LEFT, RIGHT = (0, 200, 255), (255, 120, 0)        # BGR: orange-ish left side, blue right side
LEFT_JOINTS = {1, 3, 5, 7, 9, 11, 13, 15}
MIN_SCORE = 0.3


def keypoints_for(names):
    found = {}
    for p in PKLS:
        for a in pickle.load(open(p, 'rb'))['annotations']:
            if a['frame_dir'] in names:
                i = int(np.argmax(a['keypoint_score'].reshape(len(a['keypoint_score']), -1).mean(1)))
                found[a['frame_dir']] = (a['keypoint'][i].astype(float), a['keypoint_score'][i].astype(float))
    return found


def video_path(name):
    hits = glob.glob(f'{CLIPS}/**/{name}.mp4', recursive=True)
    if not hits:
        sys.exit(f'no video found for {name} under {CLIPS}')
    return hits[0]


def draw(frame, kp, sc):
    for a, b in EDGES:
        if sc[a] >= MIN_SCORE and sc[b] >= MIN_SCORE:
            color = LEFT if a in LEFT_JOINTS and b in LEFT_JOINTS else RIGHT if a not in LEFT_JOINTS and b not in LEFT_JOINTS else (255, 255, 255)
            cv2.line(frame, tuple(map(int, kp[a])), tuple(map(int, kp[b])), color, 4, cv2.LINE_AA)
    for j in range(17):
        if sc[j] >= MIN_SCORE:
            cv2.circle(frame, tuple(map(int, kp[j])), 6, (255, 255, 255), -1, cv2.LINE_AA)


def banner(frame, text, ok):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 110, 0) if ok else (0, 0, 170), -1)
    cv2.putText(frame, text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    names = sys.argv[1:3] if len(sys.argv) >= 3 else ['EE04_Normal_back_0250-0260', 'EE04_Normal_back_0260-0270']
    kps = keypoints_for(set(names))
    caps = [cv2.VideoCapture(video_path(n)) for n in names]
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    tmp = os.path.join(OUT_DIR, '_overlay_tmp.avi')
    out = os.path.join(OUT_DIR, f'keypoint_overlay_{names[0]}_vs_{names[1][-9:]}.mp4')
    writer, stills, t = None, [], 0
    while True:
        frames = [c.read() for c in caps]
        if not all(ok for ok, _ in frames):
            break
        panels = []
        for (ok, fr), n in zip(frames, names):
            if n in kps and t < len(kps[n][0]):
                draw(fr, kps[n][0][t], kps[n][1][t])
                banner(fr, f'{n}  -  HRNet keypoints in pkl', True)
            else:
                banner(fr, f'{n}  -  NO keypoints in pkl', False)
            panels.append(fr)
        h = min(p.shape[0] for p in panels)
        combo = np.hstack([cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) for p in panels])
        if writer is None:
            writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'MJPG'), fps, (combo.shape[1], combo.shape[0]))
        writer.write(combo)
        if t in (0, 100, 200, 299):
            stills.append(cv2.resize(combo, (combo.shape[1] // 2, combo.shape[0] // 2)))
        t += 1
    writer.release()
    subprocess.run([FFMPEG, '-y', '-loglevel', 'error', '-i', tmp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out], check=True)
    os.remove(tmp)
    sheet = out.replace('.mp4', '_frames.png')
    cv2.imwrite(sheet, np.vstack(stills))
    print(f'wrote {out}\nwrote {sheet}   ({t} frames)')


if __name__ == '__main__':
    main()
