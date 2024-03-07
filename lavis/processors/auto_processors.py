from transformers import AutoImageProcessor
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor


@registry.register_processor("auto_image_eval")
class AutoImageEvalProcessor(BaseProcessor):
    def __init__(self, pretrained_model_name):
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
        assert self.processor.model_input_names is not None and self.processor.model_input_names == ["pixel_values"], \
                "Need to check that collater stacks dimensions of all inputs as expected by the model"

    def __call__(self, image):
        proccessed = self.processor(images=image, return_tensors="pt")
        for name in self.processor.model_input_names:
            proccessed[name] = proccessed[name].squeeze(0)
        return proccessed

    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name = cfg.pretrained_model_name
        return cls(pretrained_model_name=pretrained_model_name)
