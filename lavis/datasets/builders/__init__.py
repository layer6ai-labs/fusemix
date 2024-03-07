"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import load_dataset_config
from lavis.datasets.builders.image_text_pair_builder import (
    COCOCaptionBuilder,
    ConceptualCaption3MBuilder,
    VGCaptionBuilder,
    SBUCaptionBuilder,
)

from lavis.datasets.builders.retrieval_builder import (
    COCORetrievalBuilder,
    Flickr30kBuilder,
)
from lavis.datasets.builders.feature_pair_builder import (
    COCOCaptionFeatDinoV2Vitg14BgeLargeBuilder,
    COCOCaptionFeatDinoV2Vitg14CohereV3Builder,
    COCOCaptionFeatDinoV2Vitg14E5LargeBuilder,
    COCORetrievalFeatDinoV2Vitg14BgeLargeBuilder,
    COCORetrievalFeatDinoV2Vitg14CohereV3Builder,
    ConceptualCaption3MFeatDinoV2Vitg14BgeLargeBuilder,
    ConceptualCaption3MFeatDinoV2Vitg14CohereV3Builder,
    ConceptualCaption3MFeatDinoV2Vitg14E5LargeBuilder,
    Flickr30kFeatDinoV2Vitg14BgeLargeBuilder,
    Flickr30kFeatDinoV2Vitg14CohereV3Builder,
    Flickr30kFeatDinoV2Vitg14E5LargeBuilder,
    SBUCaptionFeatDinoV2Vitg14BgeLargeBuilder,
    SBUCaptionFeatDinoV2Vitg14CohereV3Builder,
    SBUCaptionFeatDinoV2Vitg14E5LargeBuilder,
    VGCaptionFeatDinoV2Vitg14BgeLargeBuilder,
    VGCaptionFeatDinoV2Vitg14CohereV3Builder,
    VGCaptionFeatDinoV2Vitg14E5LargeBuilder,
)

from lavis.common.registry import registry

__all__ = [
    "COCOCaptionBuilder",
    "COCOCaptionFeatDinoV2Vitg14BgeLargeBuilder",
    "COCOCaptionFeatDinoV2Vitg14CohereV3Builder",
    "COCOCaptionFeatDinoV2Vitg14E5LargeBuilder",
    "COCORetrievalBuilder",
    "COCORetrievalFeatDinoV2Vitg14BgeLargeBuilder",
    "COCORetrievalFeatDinoV2Vitg14CohereV3Builder",
    "ConceptualCaption3MBuilder",
    "ConceptualCaption3MFeatDinoV2Vitg14BgeLargeBuilder",
    "ConceptualCaption3MFeatDinoV2Vitg14CohereV3Builder",
    "ConceptualCaption3MFeatDinoV2Vitg14E5LargeBuilder",
    "Flickr30kBuilder",
    "Flickr30kFeatDinoV2Vitg14BgeLargeBuilder",
    "Flickr30kFeatDinoV2Vitg14CohereV3Builder",
    "Flickr30kFeatDinoV2Vitg14E5LargeBuilder",
    "SBUCaptionBuilder",
    "SBUCaptionFeatDinoV2Vitg14BgeLargeBuilder",
    "SBUCaptionFeatDinoV2Vitg14CohereV3Builder",
    "SBUCaptionFeatDinoV2Vitg14E5LargeBuilder",
    "VGCaptionBuilder",
    "VGCaptionFeatDinoV2Vitg14BgeLargeBuilder",
    "VGCaptionFeatDinoV2Vitg14CohereV3Builder",
    "VGCaptionFeatDinoV2Vitg14E5LargeBuilder",
]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())


dataset_zoo = DatasetZoo()
