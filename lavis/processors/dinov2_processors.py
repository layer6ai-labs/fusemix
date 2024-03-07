from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class DinoV2ImageBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.485, 0.456, 0.406) 
        if std is None:
            std = (0.229, 0.224, 0.225) 

        self.normalize = transforms.Normalize(mean, std)


@registry.register_processor("dinov2_image_eval")
class DinoV2ImageEvalProcessor(DinoV2ImageBaseProcessor):
    def __init__(self, image_size=224, mean=None, std=None):

        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    @classmethod
    def from_config(cls, cfg):
        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
        )
