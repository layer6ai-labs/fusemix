import torch
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


@registry.register_processor("feat_train")
class FeatTrainProcessor(BaseProcessor):
    def __init__(self, noise_std=0):
        self.noise_std = noise_std

    def __call__(self, embed):
        noise = torch.rand_like(embed)
        return embed + (self.noise_std * noise) 

    @classmethod
    def from_config(cls, cfg):
        noise_std = cfg.get("noise_std", 0)
        return cls(noise_std=noise_std)


@registry.register_processor("feat_eval")
class FeatEvalProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
