from lavis.models.text_models.base_text_model import BaseTextModel
from lavis.common.registry import registry


@registry.register_model("bge_feature_extractor")
class BgeFeatureExtractor(BaseTextModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "large": "configs/models/text_models/bge_feature_extractor_large.yaml",
    }

    def extract_features(self, samples):
        text = samples["text_input"]
        # tokenize
        input_token = self.tokenizer(text,
                                     return_tensors="pt",
                                     padding=True,
                                     truncation=True
                                     ).to(self.device)
        # forward
        output = self.model(**input_token)

        # [cls] embedding
        return {"embeds": output[0][:, 0]}
