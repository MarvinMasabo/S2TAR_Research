# Handoff file — Task: pick a transformer baseline for skeleton→energy-expenditure regression

## Context to paste into a fresh AI chat
We run skeleton-based **regression** (not classification): input = 2D COCO-17 keypoints over ~300 frames,
target = metabolic power in **Watts/kg**. Data is in pyskl `PoseDataset` pkl format
(`annotations` list with `keypoint (M,T,V,C)`, `label` float; `split` dict). Current baselines:
STGCN (test MAE 0.37–0.41 W/kg, R² 0.90–0.91) and PoseC3D (weaker, overfits).
We want to add **one transformer baseline** that is (a) one of the top performers on skeleton
action-recognition benchmarks and (b) fast for us to adapt: swap in our pkl, swap the classification
head for a 1-unit regression head + MSE loss, train.

## Task
Evaluate **SkateFormer** and **MotionBERT** on those two criteria and recommend one. Deliver:
- accuracy vs the field (NTU-60/120 X-Sub/X-View, from the survey paper we use)
- what it takes to run on our data: dataloader format, head swap, loss, 2D vs 3D input, joint count, config style
- rough effort estimate (hours) and any blockers (3D-only input, custom CUDA ops, licence)

## What we already know (2026-09)

### SkateFormer  (ECCV 2024, KAIST-VICLab)
- Repo: https://github.com/KAIST-VICLab/SkateFormer — official PyTorch, ~140 stars, HF checkpoints.
- **SOTA** on NTU RGB+D 60/120 and NW-UCLA; top of the survey table (~97.8 X-Sub120 class in the paper).
- PyTorch ≥ 1.12.1, Python ≥ 3.9 — **matches our `pyskl_310` env**.
- Input tensor `[B, C, T, V, M]`, **V=24 joints, M=2 (joint+bone branch)**, 64-frame resample, "skeletal
  partitioning" preprocessing in `./data/`. Config-driven via YAML in `./config/`.
- Head: linear logits → classes. Swap for `nn.Linear(d, 1)` + `MSELoss`; drop softmax/label smoothing.
- Licence: **research/education only** (fine for a paper, not commercial).
- Adaptation cost: **medium** — need a 24-joint layout (our data is COCO-17; either remap/pad to their
  layout or edit their graph/partition config to 17), write a small pkl→their-format dataset adapter,
  swap head+loss. No exotic CUDA ops noted.

### MotionBERT  (ICCV 2023, Walter0807)
- Repo: https://github.com/Walter0807/MotionBERT — official PyTorch, widely used.
- DSTformer encoder **pretrained** on 2D→3D motion; explicitly designed to finetune to downstream tasks
  "with a simple 1–2 layer regression head" — mesh/pose/action are all done this way in the repo.
- This is the model our reference survey/`motionbert` notes already lean on.
- Input: 2D keypoints (their pipeline uses H36M 17-joint / COCO-17) — **no joint remap needed**, closest
  to our data of the two.
- Adaptation cost: **low–medium** — `configs/` yaml, `lib/data/` dataset classes to mirror, replace the
  task head with `Linear(dim,1)`; can start from their released pretrained encoder (transfer learning,
  which Ricardo mentioned wanting to try).

### CTR-GCN  (2021, channel-wise topology refinement GCN)
- Already built into pyskl (`configs/ctrgcn/`) — no new repo, no adapter needed, same pkl/config style
  as our STGCN/DGSTGCN runs. Not a transformer, but flagged by Ricardo for the **next paper**: it learns
  an independent graph per feature channel, so with richer joint-angle input (the new 85-point capture,
  not just x/y) it can learn how e.g. hip angle changes reshape the knee's x/y — DGSTGCN can't do that.
  No action needed now; just don't forget it when the angle-based dataset is ready.

## Decision (updated 2026-09-11, team meeting)
**Recommendation flipped to SkateFormer.** MotionBERT was the natural first pick — 2D COCO-17 input,
no joint remap, and its pretrained encoder is built for exactly this kind of regression finetuning. But
Ricardo and CM already tested this transfer-learning path on the reference dataset: fine-tuning
MotionBERT's pretrained weights (trained on varied general/exercise motion) onto our narrow, walking-only,
prosthetic-inclusive domain **made results worse**, not better — a classic domain-transfer failure, not a
bug. Their reference paper reports ~1.3 error using this approach; CM's own from-scratch runs on the same
data got 0.4–0.5, i.e. skipping the mismatched pretrained weights and training task-specific from scratch
did better than transfer learning here.

That points at **SkateFormer** instead: built specifically for skeleton action recognition (not a general
motion encoder), documented Hugging Face + config-driven adaptation path, and nothing to "un-mismatch" — no
foreign pretraining domain to fight. Cost is the 24-joint layout (ours is COCO-17) and a small dataset
adapter; both mechanical, not a modeling risk like MotionBERT's domain gap.

## Next steps
1. Clone SkateFormer into `~/summer26Research/third_party/skateformer`; run its NTU demo to confirm the env
   (PyTorch ≥1.12.1 already matches `pyskl_310`).
2. Decide 17→24 joint handling: remap/pad our COCO-17 keypoints into their 24-joint layout, or edit their
   graph/partition config down to 17 (cheaper, avoids inventing joints).
3. Write `pkl → SkateFormer dataset` adapter (mirror their `feeder/` class); swap the classification head
   for `nn.Linear(d, 1)` + `MSELoss`, drop softmax/label smoothing.
4. Train from scratch (no pretrained-weight transfer, per the finding above) on `ablebody2_wattkg/exact.pkl`;
   log train/val/test MSE+MAE; add a row to `results_table.ipynb` `RUNS`.

Sources: SkateFormer https://github.com/KAIST-VICLab/SkateFormer ·
arXiv https://arxiv.org/html/2403.09508v3 ·
MotionBERT https://github.com/Walter0807/MotionBERT · https://motionbert.github.io/
