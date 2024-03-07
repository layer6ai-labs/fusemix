import os
from torch.utils.data import Subset
from torch.utils.data.dataloader import default_collate
from lavis.common.utils import get_cache_path
from lavis.common.registry import registry
from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.feature_pair_dataset import FeaturePairDataset
from lavis.datasets.datasets.feature_pair_retrieval_dataset import FeaturePairRetrievalDataset


class FeaturePairBuilder(BaseDatasetBuilder):
    train_dataset_cls = FeaturePairDataset
    eval_dataset_cls = None

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        feat_paths = build_info.get(self.data_type).storage

        datasets = dict()
        for split in feat_paths.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"]
                if is_train
                else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            split_feat_paths = feat_paths.get(split)

            split_feat_path_vis = split_feat_paths.vis
            if not os.path.isabs(split_feat_path_vis):
                split_feat_path_vis = get_cache_path(split_feat_path_vis)
            assert os.path.exists(split_feat_path_vis), "visual features storage path {} does not exist.".format(split_feat_path_vis)

            split_feat_path_text = split_feat_paths.text
            if not os.path.isabs(split_feat_path_text):
                split_feat_path_text = get_cache_path(split_feat_path_text)
            assert os.path.exists(split_feat_path_text), "text features storage path {} does not exist.".format(split_feat_path_text)

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            dataset = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                vis_feat_path=split_feat_path_vis,
                text_feat_path=split_feat_path_text,
            )

            subsample_mode = build_info.get("subsample_mode")
            if subsample_mode is not None:
                assert is_train, "subsampling only supported for training"
                assert len(dataset) <= 100000, "dataset is too big for subsampling"

                # need to bind this method
                def collater(self, samples):
                    return default_collate(samples)
                Subset.collater = collater

                subsample_size = build_info.get("subsample_size", 10000)
                if subsample_mode == "kdpp_vis":
                    indices = dataset.sample_vis_k_dpp_subset(subsample_size)
                    dataset = Subset(dataset, indices)
                elif subsample_mode == "kdpp_text":
                    indices = dataset.sample_text_k_dpp_subset(subsample_size)
                    dataset = Subset(dataset, indices)
                elif subsample_mode == "unif":
                    indices = dataset.sample_uniform_subset(subsample_size)
                    dataset = Subset(dataset, indices)
                else:
                    raise NotImplementedError

            datasets[split] = dataset

        return datasets



@registry.register_builder("coco_caption_feat_dinov2_vitg14_bge_large")
class COCOCaptionFeatDinoV2Vitg14BgeLargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap_feat_dinov2_vitg14_bge_large.yaml",
    }

@registry.register_builder("coco_caption_feat_dinov2_vitg14_cohere_v3")
class COCOCaptionFeatDinoV2Vitg14CohereV3Builder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap_feat_dinov2_vitg14_cohere_v3.yaml",
    }

@registry.register_builder("coco_caption_feat_dinov2_vitg14_e5_large")
class COCOCaptionFeatDinoV2Vitg14E5LargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_cap_feat_dinov2_vitg14_e5_large.yaml",
    }

@registry.register_builder("coco_retrieval_feat_dinov2_vitg14_bge_large")
class COCORetrievalFeatDinoV2Vitg14BgeLargeBuilder(FeaturePairBuilder):
    eval_dataset_cls = FeaturePairRetrievalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_ret_feat_dinov2_vitg14_bge_large.yaml",
    }

@registry.register_builder("coco_retrieval_feat_dinov2_vitg14_cohere_v3")
class COCORetrievalFeatDinoV2Vitg14CohereV3Builder(FeaturePairBuilder):
    eval_dataset_cls = FeaturePairRetrievalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_ret_feat_dinov2_vitg14_cohere_v3.yaml",
    }

@registry.register_builder("conceptual_caption_3m_feat_dinov2_vitg14_bge_large")
class ConceptualCaption3MFeatDinoV2Vitg14BgeLargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m_feat_dinov2_vitg14_bge_large.yaml",
    }

@registry.register_builder("conceptual_caption_3m_feat_dinov2_vitg14_cohere_v3")
class ConceptualCaption3MFeatDinoV2Vitg14CohereV3Builder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m_feat_dinov2_vitg14_cohere_v3.yaml",
    }

@registry.register_builder("conceptual_caption_3m_feat_dinov2_vitg14_e5_large")
class ConceptualCaption3MFeatDinoV2Vitg14E5LargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/conceptual_caption/defaults_3m_feat_dinov2_vitg14_e5_large.yaml",
    }

@registry.register_builder("flickr30k_feat_dinov2_vitg14_bge_large")
class Flickr30kFeatDinoV2Vitg14BgeLargeBuilder(FeaturePairBuilder):
    eval_dataset_cls = FeaturePairRetrievalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_feat_dinov2_vitg14_bge_large.yaml",
    }

@registry.register_builder("flickr30k_feat_dinov2_vitg14_cohere_v3")
class Flickr30kFeatDinoV2Vitg14CohereV3Builder(FeaturePairBuilder):
    eval_dataset_cls = FeaturePairRetrievalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_feat_dinov2_vitg14_cohere_v3.yaml",
    }

@registry.register_builder("flickr30k_feat_dinov2_vitg14_e5_large")
class Flickr30kFeatDinoV2Vitg14E5LargeBuilder(FeaturePairBuilder):
    eval_dataset_cls = FeaturePairRetrievalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/flickr30k/defaults_feat_dinov2_vitg14_e5_large.yaml",
    }

@registry.register_builder("sbu_caption_feat_dinov2_vitg14_bge_large")
class SBUCaptionFeatDinoV2Vitg14BgeLargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sbu_caption/defaults_feat_dinov2_vitg14_bge_large.yaml",
    }

@registry.register_builder("sbu_caption_feat_dinov2_vitg14_cohere_v3")
class SBUCaptionFeatDinoV2Vitg14CohereV3Builder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sbu_caption/defaults_feat_dinov2_vitg14_cohere_v3.yaml",
    }

@registry.register_builder("sbu_caption_feat_dinov2_vitg14_e5_large")
class SBUCaptionFeatDinoV2Vitg14E5LargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/sbu_caption/defaults_feat_dinov2_vitg14_e5_large.yaml",
    }

@registry.register_builder("vg_caption_feat_dinov2_vitg14_bge_large")
class VGCaptionFeatDinoV2Vitg14BgeLargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_caption_feat_dinov2_vitg14_bge_large.yaml",
    }

@registry.register_builder("vg_caption_feat_dinov2_vitg14_cohere_v3")
class VGCaptionFeatDinoV2Vitg14CohereV3Builder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_caption_feat_dinov2_vitg14_cohere_v3.yaml",
    }

@registry.register_builder("vg_caption_feat_dinov2_vitg14_e5_large")
class VGCaptionFeatDinoV2Vitg14E5LargeBuilder(FeaturePairBuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vg/defaults_caption_feat_dinov2_vitg14_e5_large.yaml",
    }