from transformers import AutoModel
from lavis.models.base_model import BaseModel


class BaseImageModel(BaseModel):
    def __init__(self, pretrained_model_name):
        super().__init__()
        self.load_model(pretrained_model_name)
    
    def load_model(self, pretrained_model_name):
        self.model = AutoModel.from_pretrained(pretrained_model_name)
    
    @classmethod
    def from_config(cls, cfg):
        pretrained_model_name = cfg.pretrained_model_name
        return cls(pretrained_model_name=pretrained_model_name)