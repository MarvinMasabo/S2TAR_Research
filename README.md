# Gait → Metabolic Power (skeleton-based energy expenditure regression)

Predicting metabolic power (Watts/kg) from 2D skeleton video clips, for able-bodied and
prosthetic-limb walking, using graph convolutional networks (STGCN, CTR-GCN) from the
[pyskl](https://github.com/kennymckormick/pyskl) toolbox.

## Data

- **Able-bodied**: `AbleBody2_full.pkl` — 22 participants, 5,874 clips, 3 walking speeds
  (Slower/Preferred/Faster), 3 camera angles (Back/Left/Right).
- **Prosthetic**: `prosthetic_wattkg/exact_merged.pkl`, built by `merge_prosthetic_sources.py` from
  Siem's TAMUSA and Thailand keypoint files plus the windowed label CSVs — 14 participants
  (3 TAMUSA + 11 Thailand), 3,239 clips. 432 clips without an Exact Windowing value are
  excluded (all of EE01, EE08 Slower/Faster, 14 single windows).
- **Label**: energy expenditure via the "Exact Windowing" method (the method the team
  settled on after comparing Exact / Nearest / Windowing-Avg-Drop across all models),
  divided by each subject's body weight → **Watts/kg**.
- **Split**: 80/10/10 train/val/test, time-ordered within each participant × speed (first 80% of
  10-s windows train, next 10% val, last 10% test); all camera views of a window stay in the
  same split. Verified with `verify_prosthetic_split.py`.

## Models

| Model | Backbone | Head | Notes |
|---|---|---|---|
| STGCN | `pyskl.models.gcns.STGCN` | `GCNHead` (1 output, MSE loss) | Fixed skeleton graph, one shared learned weight per edge |
| CTR-GCN | `pyskl.models.gcns.CTRGCN` | `GCNHead` (1 output, MSE loss) | Same base graph, but computes a separate adjacency refinement **per channel**, recomputed from the input every forward pass; requires `num_person=1` (see below) |

Both are repurposed for **regression**, not the classification task pyskl was built for:
`num_classes=1`, `loss_cls=MSELoss`, and `test_cfg=dict(average_clips='score')` (required —
without it, `GCNHead`'s predictions collapse to a constant).

### STGCN vs. CTR-GCN config differences
Only three things differ between their configs (everything else — data, optimizer, schedule — is identical):
1. `backbone.type`: `'STGCN'` vs `'CTRGCN'`
2. `graph_cfg.mode`: `'stgcn_spatial'` vs `'spatial'` (different neighbor-grouping preset, same 17-joint graph)
3. `backbone.num_person`: unset for STGCN (person-count agnostic by default) vs **must be `1`** for CTR-GCN, which always sizes its input normalization as `num_person × channels × joints` (defaults to 2, built for NTU's multi-person clips) — left at the default it crashes on the first real batch.

## Training methodology

- Optimizer: SGD, lr 0.01, momentum 0.9, nesterov, weight decay 1e-4, cosine annealing, 100 epochs.
- Validated every epoch (`--validate`); the epoch with the **lowest validation MAE** is selected as "best" (not the last epoch).
- Final reported numbers (MAE/MSE/RMSE/MRE/Pearson r/R²) come from loading *only* that best-epoch checkpoint and running a real forward pass over train/val/test — not from the training log.
- Confirmed for STGCN and CTR-GCN: after the "best" epoch, validation performance plateaus (or gets noisier) while training performance keeps improving — a healthy convergence signature, not harmful overfitting (see the `*_curve*.png` plots).

## Results (Exact Windowing, Watts/kg, test split)

| Population | Model | Best epoch | Test MAE | Test MSE | Test RMSE | Test MRE | Test r | Test R² |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| Able-bodied | STGCN | 33 | 0.390 | 0.266 | 0.516 | 9.2% | 0.950 | 0.900 |
| Able-bodied | CTR-GCN | 16 | 0.376 | 0.243 | 0.493 | 8.6% | 0.954 | 0.908 |
| Prosthetic (TAMUSA + Thailand) | STGCN | 75 | 0.216 | 0.073 | 0.270 | 6.2% | 0.910 | 0.828 |
| Prosthetic (TAMUSA + Thailand) | CTR-GCN | 38 | 0.232 | 0.079 | 0.281 | 6.7% | 0.904 | 0.813 |

**Reading it:** the two models perform comparably; CTR-GCN edges out STGCN on able-bodied data,
STGCN edges out CTR-GCN on prosthetic data. Prosthetic MAE looks smaller in absolute terms, but that's expected,
not "better" — prosthetic users walk more slowly on average, so the label range itself is
narrower (1.9–5.6 W/kg vs. ~1.4–12.7 W/kg for able-bodied), which mechanically shrinks
MAE/RMSE. Relative error (MRE) is the fairer way to compare across populations.

## Repository layout

```
build_method_pkls.py            # able-bodied: build per-method (nearest/exact/windowing) W/kg pkls
build_prosthetic_pkl.py         # prosthetic (single source): build the exact-windowing W/kg pkl
merge_prosthetic_sources.py     # merge TAMUSA + Thailand, relabel to Exact Windowing W/kg, split by moment
verify_prosthetic_split.py      # check a pkl's 80/10/10 split is stratified by participant/modality/angle

configs/stgcn/…, configs/ctrgcn/…   # per-model, per-population, per-method training configs
train_*.sh                          # launchers for the pipeline (tools/dist_train.sh + a config file)
gcn_eval.py                         # shared evaluation/plotting code used by every report_*.py script
report_stgcn_ablebody.py, report_ctrgcn_ablebody.py, report_prosthetic.py
                                     # load the real config a run used, plot its curve, report
                                     # train/val/test at the best epoch

results_table.ipynb, make_deliverables.py         # earlier multi-model comparison tooling
quick_inference.ipynb                             # first, single-checkpoint inference notebook

Gait Metabolic Results/         # organized deliverables: per population, per model, plots + CSV reports
```

## Reproducing a result

**See [`PIPELINE.md`](PIPELINE.md) for the full walkthrough** (config anatomy, auto-resume,
how to extend to a new model or dataset, and the pitfalls already hit once so they don't
need re-discovering).

```bash
# 1. Build the target pkl (able-bodied, all 3 methods)
python build_method_pkls.py

# 2. Train — auto-resumes on its own if interrupted, so it's safe to nohup and walk away
bash train_stgcn_exact_100ep.sh          # or train_ctrgcn_exact_100ep.sh

# 3. Generate the train-vs-val plot + best-epoch train/val/test report
python report_stgcn_ablebody.py          # or report_ctrgcn_ablebody.py / report_prosthetic.py
```
