from dataclasses import dataclass
from typing import Optional
import torch
from transformers.modeling_outputs import ModelOutput

@dataclass
class FusionOutputFeatures(ModelOutput):
    vis_embeds: Optional[torch.FloatTensor] = None
    vis_embeds_proj: Optional[torch.FloatTensor] = None

    text_embeds: Optional[torch.FloatTensor] = None
    text_embeds_proj: Optional[torch.FloatTensor] = None


@dataclass
class FusionOutput(ModelOutput):
    intermediate_output: Optional[FusionOutputFeatures] = None

    loss: Optional[torch.FloatTensor] = None

