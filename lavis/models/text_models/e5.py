from lavis.models.text_models.base_text_model import BaseTextModel
from lavis.common.registry import registry


@registry.register_model("e5_feature_extractor")
class E5FeatureExtractor(BaseTextModel):

    PRETRAINED_MODEL_CONFIG_DICT = {
        "large": "configs/models/text_models/e5_feature_extractor_large.yaml",
    }

    @staticmethod
    def average_pool(last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def extract_features(self, samples):
        text = samples["text_input"]
        # tokenize
        input_token = self.tokenizer(text,
                                     return_tensors="pt",
                                     max_length=512,
                                     padding=True,
                                     truncation=True
                                     ).to(self.device)
        # forward
        output = self.model(**input_token)
        embedding = self.average_pool(output.last_hidden_state, input_token['attention_mask'])

        # [cls] embedding
        return {"embeds": embedding}
