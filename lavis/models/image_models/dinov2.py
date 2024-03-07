from lavis.models.image_models.base_image_model import BaseImageModel
from lavis.common.registry import registry
import torch


@registry.register_model("dinov2_feature_extractor")
class DinoV2FeatureExtractor(BaseImageModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "vits14": "configs/models/image_models/dinov2_feature_extractor_vits14.yaml",
        "vitb14": "configs/models/image_models/dinov2_feature_extractor_vitb14.yaml",
        "vitl14": "configs/models/image_models/dinov2_feature_extractor_vitl14.yaml",
        "vitg14": "configs/models/image_models/dinov2_feature_extractor_vitg14.yaml",
    }
    
    def load_model(self, pretrained_model_name):
        self.model = torch.hub.load('facebookresearch/dinov2', pretrained_model_name)

    def extract_features(self, samples):
        image = samples["image"]
        output = self.model(image)

        # vis embeds
        return {"embeds": output}
