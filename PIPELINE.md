# The regular pipeline — the standard way to train and reproduce results

This is the only workflow used in this project. An earlier single-file "self-contained"
alternative (`train_and_plot_gcn.py`) was built to validate that this pipeline's results were
reproducible, confirmed that they were, and has since been removed — everything below is the
one path going forward.

## The three pieces, and how they fit together

```
train_<model>_<dataset>.sh          (shell launcher, one line, project root)
        │  calls
tools/dist_train.sh CONFIG GPUS ...     (pyskl, sets up torch.distributed.launch)
        │  calls
tools/train.py CONFIG --launcher pytorch ...   (pyskl, the actual training entry point)
        │  reads
configs/<model>/<dataset>/<name>.py     (one file, defines model + data + optimizer + schedule)
        │  writes
work_dirs/<model>/<dataset>/<name>/     (checkpoints, logs — one per training run)
        │  read by
report_<...>.py                         (separate script: parses the log, plots the curve,
                                          loads the best checkpoint, runs the final eval)
```

Nothing here is custom code except the `train_*.sh` launcher (a one-line convenience wrapper)
and the `report_*.py` scripts (ours, built to read pyskl's log/checkpoint output). Everything
in the middle — `tools/dist_train.sh`, `tools/train.py` — is pyskl's own maintained code, used
as-is.

## The full lifecycle, step by step

### 1. Build the data pkl
```bash
python build_method_pkls.py        # able-bodied: nearest/exact/windowing, Watts/kg
python build_prosthetic_pkl.py     # prosthetic (one source), Watts/kg
python merge_prosthetic_sources.py # merge TAMUSA + Thailand prosthetic data, split by moment
```
Each of these writes one pkl with `annotations` (clips + labels + metadata) and `split`
(`train`/`val`/`test` frame_dir lists). Verify a new split with `verify_prosthetic_split.py`
before training on it — it checks 80/10/10 stratified by participant, modality, and angle.

### 2. Write (or clone) a config file
One file, e.g. `configs/stgcn/ablebody2_wattkg/exact_100ep.py`:

```python
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(
        type='GCNHead',
        num_classes=1,                                  # 1 = regression, not classification
        in_channels=256,
        loss_cls=dict(type='MSELoss', loss_weight=1.0)), # regression loss, not cross-entropy
    test_cfg=dict(average_clips='score'))   # REQUIRED or GCNHead predictions collapse to a constant

dataset_type = 'PoseDataset'
ann_file = '/home/.../ablebody2_wattkg/exact.pkl'        # <- the pkl from step 1
train_pipeline = [ ... ]   # random 100-frame window each epoch (light augmentation)
val_pipeline   = [ ... ]   # one fixed 100-frame window (deterministic)
test_pipeline  = [ ... ]   # 10 windows, averaged (steadier final number)

data = dict(
    videos_per_gpu=16, workers_per_gpu=2,
    train=dict(type='RepeatDataset', times=5,           # loop train set 5x/epoch
               dataset=dict(type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline)),
    val=dict(type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='test', pipeline=test_pipeline))

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 100
checkpoint_config = dict(interval=1)               # save every epoch
evaluation = dict(
    interval=1,                                    # validate every epoch -> full curve later
    metrics=['mean_squared_error', 'mean_absolute_error'],
    save_best='mean_absolute_error', rule='less')  # tracks the best checkpoint automatically
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
log_level = 'INFO'
work_dir = './work_dirs/stgcn/ablebody2_wattkg/exact_100ep'   # <- everything lands here
```

Every section, what it controls:
- **`model`** — architecture, loss, and the `test_cfg` fix that keeps GCNHead from collapsing.
- **`ann_file` / `data`** — which pkl, which pipeline (train gets light augmentation + repeats; val/test are deterministic).
- **`optimizer` / `lr_config`** — SGD + cosine decay, the recipe already proven stable on this data. (Don't blindly copy a borrowed template's own tuned LR — see Pitfall #2 below.)
- **`evaluation.save_best`** — this is what makes "best checkpoint" a real, log-derived fact rather than a guess: pyskl validates every epoch and remembers which one had the lowest MAE.
- **`work_dir`** — the one thing that must be unique per run; everything else (logs, checkpoints, later the report) is keyed off this path.

### 3. Train
```bash
bash train_stgcn_exact_100ep.sh
# = bash tools/dist_train.sh configs/stgcn/ablebody2_wattkg/exact_100ep.py 1 \
#       --validate --test-last --test-best --seed 0
```
- `--validate` — evaluate on val after every epoch (needed for `save_best` and the curve).
- `--test-last` / `--test-best` — pyskl's own official test-set numbers, printed at the end, for a quick independent sanity check before the full `report_*.py` run.
- **Auto-resume is automatic**: if `work_dir/latest.pth` already exists (e.g. the run was killed and restarted), `tools/train.py` picks it up and continues from there — no flag needed, nothing to remember.

Run it under `nohup ... &` (or tmux) so it survives a disconnect:
```bash
nohup bash train_stgcn_exact_100ep.sh > stgcn_run.out 2>&1 &
```

### 4. Generate the report + plot
A separate, small script — `report_stgcn_ablebody.py`, `report_ctrgcn_ablebody.py`, or
`report_prosthetic.py` — does this. Every one of them **loads the real config file the run
actually used** (`Config.fromfile(...)`), so a report can never drift out of sync with what
was actually trained, and every one of them calls the same shared functions from `gcn_eval.py`
so every model/population is scored identically:
```python
cfg = Config.fromfile('configs/stgcn/ablebody2_wattkg/exact_100ep.py')  # the real config, not a copy
logs = glob.glob(os.path.join(cfg.work_dir, '*.log'))

tr = ge.epoch_train_mse(logs)            # per-epoch train MSE, read from the log (no re-inference)
vmse, vmae = ge.epoch_val_metrics(logs)  # per-epoch val MSE/MAE, read from the log
best_ep = min(vmae, key=vmae.get)        # lowest val MAE = "best epoch"
ge.plot_train_vs_val(tr, vmse, vmae, best_ep, cfg.total_epochs, title, out_png)

# then, ONE real inference pass, only at best_ep:
model = build_model(cfg.model); load_checkpoint(model, f'{cfg.work_dir}/epoch_{best_ep}.pth')
for split in ['train', 'val', 'test']:
    p, y = ge.run_inference(model, ge.build_eval_loader(cfg, split))
    ge.metrics(p, y)   # MAE, MSE, RMSE, r, R2
```
To point this at a new run, write a new `report_<model>_<dataset>.py` that loads a different
config path — the shared functions in `gcn_eval.py` (`epoch_train_mse`, `epoch_val_metrics`,
`run_inference`, `metrics`, `build_eval_loader`, `plot_train_vs_val`) never need to change.

## Extending to a new model (e.g. SkateFormer)

1. Find or write a base config for the model in pyskl's `configs/<model>/` (look for one already
   using 2D COCO-17 + HRNet keypoints if possible — closest starting point).
2. Clone it into `configs/<model>/<dataset>_wattkg/exact_100ep.py`, and adapt:
   - `cls_head`: `num_classes` → 1, add `loss_cls=dict(type='MSELoss', loss_weight=1.0)`, add `test_cfg=dict(average_clips='score')`.
   - `ann_file` → your Watts/kg pkl; `split` values → `'train'/'val'/'test'` (not whatever the source benchmark used, e.g. NTU's `'xsub_train'`).
   - `work_dir` → a new, unique path.
3. **Check the backbone's own defaults before trusting them** — CTR-GCN's `num_person` default (2, silently wrong for single-person data) and its own template's `lr=0.1` (silently unstable on a small regression set) both had to be caught and fixed by hand. Do a quick forward-pass smoke test (build the dataset, build the model, run one batch through it) before committing to a multi-hour training run.
4. Write `train_<model>_<dataset>.sh` (copy an existing one, change the config path).
5. Copy an existing `report_*.py` and change only the `CONFIG` path it loads — every function
   it calls comes from `gcn_eval.py` and needs no changes.

## Extending to a new population/dataset (e.g. combined able-bodied + prosthetic)

1. Build the merged pkl (`merge_prosthetic_sources.py` is the pattern). Split by **moment**: every camera
   view of the same time window must land in the same split, or test answers leak into training.
2. Verify it (`verify_prosthetic_split.py`).
3. Clone an existing config, point `ann_file` at the new pkl, give it a new `work_dir`.
4. Train, resume-safe, exactly as in step 3 above.

## Pitfalls already hit once — don't re-discover these

1. **`test_cfg=dict(average_clips='score')` is required on every GCNHead config.** Without it, predictions collapse to a constant.
2. **Don't blindly reuse a borrowed template's learning rate.** PoseC3D's and CTR-GCN's own NTU-benchmark templates used `lr=0.05`/`0.1`, tuned for a much larger classification dataset — both caused the validation curve to oscillate instead of converge on our much smaller regression data. `lr=0.01` (STGCN's proven-stable value) was reused for both instead.
3. **CTR-GCN's `num_person` defaults to 2** (built for NTU's multi-person clips) and must be set to `1` for our single-person data, or the model crashes on the first real batch with a batchnorm channel mismatch.
4. **A pkl pickled under numpy≥2.0 won't load under this project's numpy<2.0 training env** (`No module named 'numpy._core'`). Fix: alias `numpy._core` to `numpy.core` before unpickling (see the top of `build_prosthetic_pkl.py` / `merge_prosthetic_sources.py`) — then re-save it, and the output is natively loadable with no special handling downstream.
5. **A DataLoader with `workers_per_gpu>0` re-imports the launching script in each worker process.** Any script used as a direct training entry point needs its real work behind `if __name__ == '__main__':` — a script without that guard will re-run its own setup code (including re-binding a distributed process group) once per worker and hang or crash.
