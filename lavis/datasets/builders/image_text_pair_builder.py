"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from lavis.common.registry import registry

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.image_text_pair_datasets import ImageTextPairDataset



@registry.register_builder("coco_caption")
class COCOCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap.yaml",
    }


@registry.register_builder("conceptual_caption_3m")
class ConceptualCaption3MBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m.yaml"
    }


@registry.register_builder("sbu_caption")
class SBUCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/sbu_caption/defaults.yaml"}


@registry.register_builder("vg_caption")
class VGCaptionBuilder(BaseDatasetBuilder):
    train_dataset_cls = ImageTextPairDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_caption.yaml"}
