from lavis.models.base_model import BaseModel
from lavis.common.registry import registry
import cohere
import torch


@registry.register_model("cohere_feature_extractor")
class CohereFeatureExtractor(BaseModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "v3": "configs/models/text_models/cohere_feature_extractor_v3.yaml",
    }

    def __init__(self, pretrained_model_name):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.co = cohere.Client("your API key")

    @property
    def device(self):
        return torch.device("cpu")

    def extract_features(self, samples):
        text = samples["text_input"]
        embeds = self.co.embed(text, input_type="search_query", model=self.pretrained_model_name).embeddings
        embeds = torch.Tensor(embeds).to(self.device)
        return {"embeds": embeds}

    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name = cfg.pretrained_model_name
        return cls(pretrained_model_name=pretrained_model_name)
