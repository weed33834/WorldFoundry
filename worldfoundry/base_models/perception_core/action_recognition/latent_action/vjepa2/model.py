import math
from functools import partial

import torch
from torch import nn
from torch.nn.init import trunc_normal_
from torchvision.transforms import functional as transform

from .modules import Block


class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size, tubelet_size, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, value):
        return self.proj(value).flatten(2).transpose(1, 2)


class _VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        in_channels=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
    ):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.init_std = init_std
        self.patch_embed = PatchEmbed3D(patch_size, tubelet_size, in_channels, embed_dim)
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                use_rope=True,
            )
            for _ in range(depth)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            trunc_normal_(module.weight, std=self.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _rescale_blocks(self):
        for layer_id, block in enumerate(self.blocks, start=1):
            scale = math.sqrt(2.0 * layer_id)
            block.attn.proj.weight.data.div_(scale)
            block.mlp.fc2.weight.data.div_(scale)

    def _tokens(self, video):
        if video.ndim != 5:
            raise ValueError(f"Expected BCTHW video tensor, got {video.shape}")
        _, _, frame_count, height, width = video.shape
        if frame_count % self.tubelet_size:
            raise ValueError(f"Frame count {frame_count} must be divisible by {self.tubelet_size}")
        return (
            self.patch_embed(video),
            frame_count // self.tubelet_size,
            height // self.patch_size,
            width // self.patch_size,
        )

    def _encode(self, value, frames, height, width):
        for block in self.blocks:
            value = block(value, frames=frames, height=height, width=width)
        return value


class VJEPA2VisionTransformer(_VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm = kwargs.get("norm_layer", nn.LayerNorm)(self.embed_dim)
        self._init_weights(self.norm)
        self._rescale_blocks()

    def forward(self, video):
        value, frames, height, width = self._tokens(video)
        return self.norm(self._encode(value, frames, height, width))


class VJEPA21VisionTransformer(_VisionTransformer):
    def __init__(self, modality_embedding=True, **kwargs):
        super().__init__(**kwargs)
        norm_layer = kwargs.get("norm_layer", nn.LayerNorm)
        self.norms_block = nn.ModuleList(norm_layer(self.embed_dim) for _ in range(4))
        self.modality_embedding = modality_embedding
        if modality_embedding:
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.normal_(self.video_mod_embed, std=1e-6)
        self.norms_block.apply(self._init_weights)
        self._rescale_blocks()

    def forward(self, video):
        value, frames, height, width = self._tokens(video)
        if self.modality_embedding:
            value = value + self.video_mod_embed
        return self.norms_block[-1](self._encode(value, frames, height, width))


class ClipAggregation(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.tubelet_size = model.tubelet_size
        self.embed_dim = model.embed_dim
        self.num_heads = model.num_heads

    def forward(self, clips, clip_indices=None):
        del clip_indices
        num_clips = len(clips)
        num_views = len(clips[0])
        batch_size, _, frame_count, _, _ = clips[0][0].shape
        flattened = [torch.cat(clip, dim=0) for clip in clips]
        output = self.model(torch.cat(flattened, dim=0))
        _, token_count, feature_dim = output.shape
        temporal_tokens = frame_count // self.tubelet_size
        spatial_tokens = token_count // temporal_tokens
        effective_batch = batch_size * num_views
        grouped = [[] for _ in range(num_views)]
        for clip_index in range(num_clips):
            clip_output = output[clip_index * effective_batch : (clip_index + 1) * effective_batch]
            for view_index in range(num_views):
                grouped[view_index].append(clip_output[view_index * batch_size : (view_index + 1) * batch_size])
        return [
            torch.cat(
                [
                    value.reshape(
                        batch_size,
                        temporal_tokens,
                        spatial_tokens,
                        feature_dim,
                    )
                    for value in values
                ],
                dim=1,
            ).flatten(1, 2)
            for values in grouped
        ]


class EvalVideoTransform:
    def __init__(
        self,
        crop_size,
        normalize=((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ):
        self.crop_size = crop_size
        self.mean, self.std = normalize

    def __call__(self, frames):
        tensors = []
        for frame in frames:
            value = transform.to_tensor(frame)
            value = transform.resize(value, self.crop_size, antialias=True)
            value = transform.center_crop(value, [self.crop_size, self.crop_size])
            tensors.append(transform.normalize(value, self.mean, self.std))
        return [torch.stack(tensors, dim=1)]


def _vit_large(version, resolution, frames):
    model_class = VJEPA21VisionTransformer if version == "2.1" else VJEPA2VisionTransformer
    return model_class(
        img_size=resolution,
        patch_size=16,
        num_frames=frames,
        tubelet_size=2,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def load_vjepa2(checkpoint, version="2", resolution=224, frames=16):
    if version not in {"2", "2.1"}:
        raise ValueError(f"Unsupported V-JEPA version: {version}")
    if not checkpoint:
        variable = "VJEPA21_CKPT_PATH" if version == "2.1" else "VJEPA2_CKPT_PATH"
        raise ValueError(f"{variable} must point to the encoder checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_key = "ema_encoder" if version == "2.1" else "target_encoder"
    source = {
        key.replace("module.", "").replace("backbone.", ""): value for key, value in payload[checkpoint_key].items()
    }
    model = _vit_large(version, resolution, frames)
    target = model.state_dict()
    model.load_state_dict(
        {key: value for key, value in source.items() if key in target and value.shape == target[key].shape},
        strict=False,
    )
    return ClipAggregation(model)
