"""In-tree Cosmos3 Diffusers inference components."""

import diffusers

from .audio_tokenizer import Cosmos3AVAEAudioTokenizer
from .pipeline import Cosmos3OmniPipeline, Cosmos3OmniPipelineOutput, CosmosActionCondition
from .transformer import Cosmos3OmniTransformer

Cosmos3OmniDiffusersPipeline = Cosmos3OmniPipeline

# Spoof the eventual upstream module paths so save_pretrained writes
# "diffusers" as the library in model_index.json.
Cosmos3OmniTransformer.__module__ = "diffusers.models.transformers.transformer_cosmos3"
Cosmos3OmniPipeline.__module__ = "diffusers.pipelines.cosmos.pipeline_cosmos3_omni"
Cosmos3AVAEAudioTokenizer.__module__ = "diffusers.models.autoencoders.autoencoder_cosmos3_audio"

# Loader does `getattr(importlib.import_module(library), class_name)`,
# so the class must be reachable as `diffusers.<ClassName>`.
diffusers.Cosmos3OmniTransformer = Cosmos3OmniTransformer
diffusers.Cosmos3OmniDiffusersPipeline = Cosmos3OmniDiffusersPipeline
diffusers.Cosmos3OmniPipeline = Cosmos3OmniPipeline
diffusers.Cosmos3AVAEAudioTokenizer = Cosmos3AVAEAudioTokenizer

__all__ = [
    "Cosmos3OmniDiffusersPipeline",
    "Cosmos3OmniPipeline",
    "Cosmos3OmniPipelineOutput",
    "Cosmos3OmniTransformer",
    "Cosmos3AVAEAudioTokenizer",
    "CosmosActionCondition",
]
