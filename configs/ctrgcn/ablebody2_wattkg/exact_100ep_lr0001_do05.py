# CTR-GCN / Able-bodied (AbleBody2) / Exact Windowing -- regularization experiment (Dr. Ramirez, 11/8 meeting).
# Identical to the OG run ./exact_100ep.py except for the two settings below:
#   learning rate  0.01 -> 0.001   (10x smaller steps: learns more slowly and steadily)
#   dropout        none -> 0.5      (during training only, randomly zero 50% of the 256 features
#                                   that feed the final W/kg layer, so the model can't lean on any one feature)
# Everything else (data file, split, 100 epochs, cosine LR schedule, batch size 16, validation every
# epoch, best checkpoint by val MAE) is inherited unchanged from the base config.
_base_ = ['./exact_100ep.py']

model = dict(cls_head=dict(dropout=0.5))
optimizer = dict(lr=0.001)
work_dir = './work_dirs/ctrgcn/ablebody2_wattkg/exact_100ep_lr0001_do05'
