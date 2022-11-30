#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import build_model, MODEL_REGISTRY  # noqa
from .custom_video_model_builder import *  # noqa
from .video_model_builder import ResNet, SlowFast  # noqa

try:
    from .ptv_model_builder import (
        PTVCSN,
        PTVR2plus1D,
        PTVResNet,
        PTVSlowFast,
        PTVX3D,
    )  # noqa
except Exception:
    print("Please update your PyTorchVideo to latest master")
