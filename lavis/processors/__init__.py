"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.processors.base_processor import BaseProcessor

from lavis.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from lavis.processors.auto_processors import AutoImageEvalProcessor
from lavis.processors.feat_processors import (
    FeatTrainProcessor,
    FeatEvalProcessor,
)
from lavis.processors.dinov2_processors import DinoV2ImageEvalProcessor

from lavis.common.registry import registry

__all__ = [
    "BaseProcessor",
    # Auto
    "AutoImageEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    # DinoV2
    "DinoV2ImageEvalProcessor",
    # Features
    "FeatTrainProcessor",
    "FeatEvalProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("blip_image_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
