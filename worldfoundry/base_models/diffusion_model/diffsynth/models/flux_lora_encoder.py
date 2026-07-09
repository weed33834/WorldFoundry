"""Module for base_models -> diffusion_model -> diffsynth -> models -> flux_lora_encoder.py functionality."""

import torch
from .sd_text_encoder import CLIPEncoderLayer


class LoRALayerBlock(torch.nn.Module):
    """Lo ra layer block implementation."""
    def __init__(self, L, dim_in, dim_out):
        """Init.

        Args:
            L: The l.
            dim_in: The dim in.
            dim_out: The dim out.
        """
        super().__init__()
        self.x = torch.nn.Parameter(torch.randn(1, L, dim_in))
        self.layer_norm = torch.nn.LayerNorm(dim_out)

    def forward(self, lora_A, lora_B):
        """Forward.

        Args:
            lora_A: The lora a.
            lora_B: The lora b.
        """
        x = self.x @ lora_A.T @ lora_B.T
        x = self.layer_norm(x)
        return x
    

class LoRAEmbedder(torch.nn.Module):
    """Lo ra embedder implementation."""
    def __init__(self, lora_patterns=None, L=1, out_dim=2048):
        """Init.

        Args:
            lora_patterns: The lora patterns.
            L: The l.
            out_dim: The out dim.
        """
        super().__init__()
        if lora_patterns is None:
            lora_patterns = self.default_lora_patterns()
            
        model_dict = {}
        for lora_pattern in lora_patterns:
            name, dim = lora_pattern["name"], lora_pattern["dim"]
            model_dict[name.replace(".", "___")] = LoRALayerBlock(L, dim[0], dim[1])
        self.model_dict = torch.nn.ModuleDict(model_dict)
        
        proj_dict = {}
        for lora_pattern in lora_patterns:
            layer_type, dim = lora_pattern["type"], lora_pattern["dim"]
            if layer_type not in proj_dict:
                proj_dict[layer_type.replace(".", "___")] = torch.nn.Linear(dim[1], out_dim)
        self.proj_dict = torch.nn.ModuleDict(proj_dict)
        
        self.lora_patterns = lora_patterns
        
        
    def default_lora_patterns(self):
        """Default lora patterns."""
        lora_patterns = []
        lora_dict = {
            "attn.a_to_qkv": (3072, 9216), "attn.a_to_out": (3072, 3072), "ff_a.0": (3072, 12288), "ff_a.2": (12288, 3072), "norm1_a.linear": (3072, 18432),
            "attn.b_to_qkv": (3072, 9216), "attn.b_to_out": (3072, 3072), "ff_b.0": (3072, 12288), "ff_b.2": (12288, 3072), "norm1_b.linear": (3072, 18432),
        }
        for i in range(19):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        lora_dict = {"to_qkv_mlp": (3072, 21504), "proj_out": (15360, 3072), "norm.linear": (3072, 9216)}
        for i in range(38):
            for suffix in lora_dict:
                lora_patterns.append({
                    "name": f"single_blocks.{i}.{suffix}",
                    "dim": lora_dict[suffix],
                    "type": suffix,
                })
        return lora_patterns
        
    def forward(self, lora):
        """Forward.

        Args:
            lora: The lora.
        """
        lora_emb = []
        for lora_pattern in self.lora_patterns:
            name, layer_type = lora_pattern["name"], lora_pattern["type"]
            lora_A = lora[name + ".lora_A.default.weight"]
            lora_B = lora[name + ".lora_B.default.weight"]
            lora_out = self.model_dict[name.replace(".", "___")](lora_A, lora_B)
            lora_out = self.proj_dict[layer_type.replace(".", "___")](lora_out)
            lora_emb.append(lora_out)
        lora_emb = torch.concat(lora_emb, dim=1)
        return lora_emb
    
    
class FluxLoRAEncoder(torch.nn.Module):
    """Flux lo ra encoder implementation."""
    def __init__(self, embed_dim=4096, encoder_intermediate_size=8192, num_encoder_layers=1, num_embeds_per_lora=16, num_special_embeds=1):
        """Init.

        Args:
            embed_dim: The embed dim.
            encoder_intermediate_size: The encoder intermediate size.
            num_encoder_layers: The num encoder layers.
            num_embeds_per_lora: The num embeds per lora.
            num_special_embeds: The num special embeds.
        """
        super().__init__()
        self.num_embeds_per_lora = num_embeds_per_lora
        # embedder
        self.embedder = LoRAEmbedder(L=num_embeds_per_lora, out_dim=embed_dim)
        
        # encoders
        self.encoders = torch.nn.ModuleList([CLIPEncoderLayer(embed_dim, encoder_intermediate_size, num_heads=32, head_dim=128) for _ in range(num_encoder_layers)])

        # special embedding
        self.special_embeds = torch.nn.Parameter(torch.randn(1, num_special_embeds, embed_dim))
        self.num_special_embeds = num_special_embeds
        
        # final layer
        self.final_layer_norm = torch.nn.LayerNorm(embed_dim)
        self.final_linear = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, lora):
        """Forward.

        Args:
            lora: The lora.
        """
        lora_embeds = self.embedder(lora)
        special_embeds = self.special_embeds.to(dtype=lora_embeds.dtype, device=lora_embeds.device)
        embeds = torch.concat([special_embeds, lora_embeds], dim=1)
        for encoder_id, encoder in enumerate(self.encoders):
            embeds = encoder(embeds)
        embeds = embeds[:, :self.num_special_embeds]
        embeds = self.final_layer_norm(embeds)
        embeds = self.final_linear(embeds)
        return embeds
    
    @staticmethod
    def state_dict_converter():
        """State dict converter."""
        return FluxLoRAEncoderStateDictConverter()


class FluxLoRAEncoderStateDictConverter:
    """Flux lo ra encoder state dict converter implementation."""
    def from_civitai(self, state_dict):
        """From civitai.

        Args:
            state_dict: The state dict.
        """
        return state_dict
