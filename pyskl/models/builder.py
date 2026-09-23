# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

# Fix for Streamlit reruns: check if registry already exists
try:
    MODELS = Registry('models', parent=MMCV_MODELS)
except AssertionError:
    # Registry already exists (happens on Streamlit rerun)
    # Find the existing registry in the parent's children
    for child in MMCV_MODELS._children:
        if child.scope == 'pyskl':
            MODELS = child
            break
    else:
        # Fallback: create without parent to avoid conflict
        MODELS = Registry('models')
BACKBONES = MODELS
HEADS = MODELS
RECOGNIZERS = MODELS
LOSSES = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_recognizer(cfg):
    """Build recognizer."""
    return RECOGNIZERS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_model(cfg):
    """Build model."""
    args = cfg.copy()
    obj_type = args.pop('type')
    if obj_type in RECOGNIZERS:
        return build_recognizer(cfg)
    raise ValueError(f'{obj_type} is not registered')
