# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> diffusion_model -> image -> sana -> diffusion -> model -> builder.py functionality."""

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderDC
from diffusers.models import AutoencoderKL
from diffusers.models.autoencoders import AutoencoderKLLTX2Video
from mmcv import Registry
from termcolor import colored
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    CLIPVisionModel,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
    T5Tokenizer,
)
from transformers import logging as transformers_logging

from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint
from worldfoundry.core.checkpoint import load_weights_only, require_mapping
from worldfoundry.core.io import hf_download_or_fpath
from worldfoundry.core.model_loading.file import load_state_dict as _load_sana_state_dict

MODELS = Registry("models")

transformers_logging.set_verbosity_error()


def _prefer_local_hf_repo(repo_id: str) -> str:
    """Helper function to prefer local hf repo.

    Args:
        repo_id: The repo id.

    Returns:
        The return value.
    """
    repo_key = repo_id.replace("/", "--")

    def is_pretrained_repo(path: Path) -> bool:
        # Encoder-only repositories expose config.json at the root, while
        # diffusers pipelines such as LTX-2 expose model_index.json and keep
        # the component config under a subfolder (``vae/config.json``).
        return (path / "config.json").is_file() or (path / "model_index.json").is_file()

    roots = []
    if os.environ.get("WORLDFOUNDRY_HFD_ROOT"):
        roots.append(Path(os.environ["WORLDFOUNDRY_HFD_ROOT"]).expanduser())
    if os.environ.get("WORLDFOUNDRY_CKPT_DIR"):
        roots.append(Path(os.environ["WORLDFOUNDRY_CKPT_DIR"]).expanduser() / "hfd")
    roots.append(Path("~/.cache/worldfoundry/checkpoints/hfd").expanduser())

    seen = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for candidate in (
            root / repo_key,
            root / "hub" / f"models--{repo_key}",
            root / f"models--{repo_key}",
        ):
            if not candidate.exists():
                continue
            snapshots = candidate / "snapshots"
            if snapshots.is_dir():
                revisions = sorted((path for path in snapshots.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
                for revision in revisions:
                    if is_pretrained_repo(revision):
                        return str(revision)
            if is_pretrained_repo(candidate):
                return str(candidate)
    return repo_id


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    """Build model.

    Args:
        cfg: The cfg.
        use_grad_checkpoint: The use grad checkpoint.
        use_fp32_attention: The use fp32 attention.
        gc_step: The gc step.
    """
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)

    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(name="T5", device="cuda"):
    """Get tokenizer and text encoder.

    Args:
        name: The name.
        device: The device.
    """
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen2-5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
    assert name in list(text_encoder_dict.keys()), f"not support this text encoder: {name}"
    model_ref = _prefer_local_hf_repo(text_encoder_dict[name])
    if "T5" in name:
        tokenizer = T5Tokenizer.from_pretrained(model_ref)
        text_encoder = T5EncoderModel.from_pretrained(model_ref, torch_dtype=torch.float16).to(device)
    elif "gemma" in name:
        tokenizer = AutoTokenizer.from_pretrained(model_ref)
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(model_ref, torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
    elif "Qwen" in name:
        from worldfoundry.base_models.llm_mllm_core.mllm.qwen.qwen_vl_embedder import QwenVLEmbedder

        text_handler = QwenVLEmbedder(model_id=model_ref, device=device)
        return None, text_handler
    else:
        print("error load text encoder")
        exit()

    return tokenizer, text_encoder


def get_image_encoder(name, model_path, tokenizer_path=None, device="cuda", dtype=None, config=None):
    """Get image encoder.

    Args:
        name: The name.
        model_path: The model path.
        tokenizer_path: The tokenizer path.
        device: The device.
        dtype: The dtype.
        config: The config.
    """
    if name == "CLIP":
        from worldfoundry.base_models.diffusion_model.video.wan.media_encoders.linear_clip import CLIPModel

        image_encoder = CLIPModel(dtype, device, model_path, tokenizer_path)
    elif name == "flux-siglip":
        image_encoder = SiglipVisionModel.from_pretrained(model_path, subfolder="image_encoder", torch_dtype=dtype).to(
            device
        )
        image_processor = SiglipImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")
        return image_encoder.eval().requires_grad_(False), image_processor
    else:
        raise ValueError(f"Unsupported image encoder: {name}")

    return image_encoder


@torch.no_grad()
def encode_image(name, image_encoder, images, device="cuda", image_processor=None, dtype=None):
    """Encode image.

    Args:
        name: The name.
        image_encoder: The image encoder.
        images: The images.
        device: The device.
        image_processor: The image processor.
        dtype: The dtype.
    """
    if image_encoder is None:
        return None
    if name == "CLIP":
        image_embeds = image_encoder.visual(images.to(image_encoder.device))
        return image_embeds.to(device, images.dtype)
    elif name == "flux-siglip":
        dtype = dtype or image_encoder.dtype
        images = (images + 1) / 2.0  # [-1, 1] -> [0, 1]
        images = image_processor(images=images.clamp(0, 1), return_tensors="pt", do_rescale=False).to(
            device=device, dtype=image_encoder.dtype
        )
        image_embeds = image_encoder(**images).last_hidden_state
        return image_embeds.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported image encoder: {name}")


def get_vae(name, model_path, device="cuda", dtype=None, config=None):
    """Get vae.

    Args:
        name: The name.
        model_path: The model path.
        device: The device.
        dtype: The dtype.
        config: The config.
    """
    if name == "sdxl" or name == "sd3":
        vae = AutoencoderKL.from_pretrained(model_path).to(device).to(torch.float16)
        if name == "sdxl":
            vae.config.shift_factor = 0
        return vae.to(dtype)
    elif ("dc-ae" in name and not "st-dc-ae" in name) or "dc-vae" in name:
        from diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF

        print(colored(f"[DC-AE] Loading model from {model_path}", attrs=["bold"]))
        dc_ae = DCAE_HF.from_pretrained(model_path).to(device).eval()
        return dc_ae.to(dtype)
    elif "st-dc-ae" in name:
        from diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAEWithTemporal_HF

        print(colored(f"[ST-DC-AE] Loading model from {model_path}", attrs=["bold"]))
        dc_ae = DCAEWithTemporal_HF.from_pretrained(model_path, model_name=name).to(device).eval()
        if config.scaling_factor is not None:
            dc_ae.cfg.scaling_factor = torch.tensor(config.scaling_factor).to(dtype).to(device)
        return dc_ae.to(dtype)

    elif "AutoencoderDC" in name:
        print(colored(f"[AutoencoderDC] Loading model from {model_path}", attrs=["bold"]))
        dc_ae = AutoencoderDC.from_pretrained(model_path).to(device).eval()
        return dc_ae.to(dtype)
    elif "WanVAE" in name:
        from worldfoundry.base_models.diffusion_model.video.wan.vae.linear_wan import WanVAE

        assert config is not None, "config.vae is required for WanVAE"
        print(colored(f"[WanVAE] Loading model from {model_path}", attrs=["bold"]))
        vae = WanVAE(
            z_dim=config.vae_latent_dim,
            vae_pth=config.vae_pretrained,
            dtype=dtype,
            device=device,
        )
        return vae
    elif "Wan2_2_VAE" in name:
        from worldfoundry.base_models.diffusion_model.video.wan.wan_2p2.modules.vae2_2 import Wan2_2_VAE

        assert config is not None, "config.vae is required for Wan2_2_VAE"
        print(colored(f"[Wan2_2_VAE] Loading model from {model_path}", attrs=["bold"]))
        vae = Wan2_2_VAE(
            z_dim=config.vae_latent_dim,
            vae_pth=config.vae_pretrained,
            dtype=dtype,
            device=device,
        )
        return vae
    elif "LTX2VAE_chunk_tile" in name:
        # Public LTX-2 VAE loaded through the in-tree causal wrapper so long
        # SANA-Streaming inference can use temporal-only chunk tiling.
        from diffusion.model.ltx2.causal_vae import AutoencoderKLCausalLTX2Video

        assert config is not None, "config.vae is required for LTX2VAE_chunk_tile"
        vae_ref = _prefer_local_hf_repo(config.vae_pretrained)
        print(colored(f"[LTX2VAE_chunk_tile] Loading model from {vae_ref}", attrs=["bold"]))
        vae = (
            AutoencoderKLCausalLTX2Video.from_pretrained(
                vae_ref,
                subfolder="vae",
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        vae.enable_tiling(tile_sample_min_num_frames=24, tile_sample_stride_num_frames=8)
        return vae
    elif "LTX2VAE_diffusers" in name:
        # Use diffusers AutoencoderKLLTX2Video for LTX2
        assert config is not None, "config.vae is required for LTX2VAE_diffusers"
        vae_ref = _prefer_local_hf_repo(config.vae_pretrained)
        print(colored(f"[LTX2VAE_diffusers] Loading model from {vae_ref}", attrs=["bold"]))
        vae = (
            AutoencoderKLLTX2Video.from_pretrained(vae_ref, subfolder="vae", torch_dtype=dtype)
            .to(device)
            .eval()
        )
        return vae
    else:
        print("error load vae")
        exit()


@torch.no_grad()
def vae_encode(name, vae, images, sample_posterior=True, device="cuda", cache_key=None, if_cache=False, data_info=None):
    """Vae encode.

    Args:
        name: The name.
        vae: The vae.
        images: The images.
        sample_posterior: The sample posterior.
        device: The device.
        cache_key: The cache key.
        if_cache: The if cache.
        data_info: The data info.
    """
    dtype = images.dtype
    if name == "sdxl" or name == "sd3":
        posterior = vae.encode(images.to(device)).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    elif "dc-ae" in name and not "st-dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor is not None else 0.41407
        z = ae.encode(images.to(device))
        z = z * scaling_factor
    elif "dc-vae" in name or "st-dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor is not None else 0.493
        if isinstance(cache_key, list) and ae.cfg.cache_dir is not None:
            cache_file = [os.path.join(ae.cfg.cache_dir, f"{key}.npz") for key in cache_key]
        else:
            cache_file = None

        z = None
        try:
            if data_info is None:
                z = torch.stack([torch.from_numpy(np.load(cf)["z"]).to(device) for cf in cache_file], dim=0)
            elif data_info is not None and data_info.get("zip_file", None) is not None:
                z = []
                for zip_file, key, dataset_name in zip(
                    data_info["zip_file"], data_info["key"], data_info["dataset_name"]
                ):
                    vae_zip_file = os.path.join(ae.cfg.cache_dir, dataset_name, os.path.basename(zip_file))
                    if os.path.exists(vae_zip_file):
                        from diffusion.data.zip_cache import open_zip_file

                        z_vae_cache = open_zip_file(vae_zip_file)
                        with z_vae_cache.open(key + ".npz", "r") as f:
                            z.append(np.load(f)["z"] if "z" in np.load(f) else np.load(f))
                z = torch.from_numpy(np.stack(z)).to(device)
        except:
            z = None
        if z is None or len(z) == 0:
            z = ae.encode(images.to(device))
            if isinstance(scaling_factor, float):
                z = z * scaling_factor
            else:
                z = z * scaling_factor[None].view(1, -1, 1, 1, 1)
            if cache_file is not None and if_cache:
                tempdir = os.path.join(ae.cfg.cache_dir, ".tmp")
                os.makedirs(tempdir, exist_ok=True, mode=0o777)
                for i, cf in enumerate(cache_file):
                    if os.path.exists(cf):
                        continue
                    os.makedirs(os.path.dirname(cf), exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=tempdir) as f:
                        np.savez_compressed(f, z=z[i].float().cpu().numpy())  # bf16 not support for cpu
                        try:
                            os.link(f.name, cf)
                        except:
                            pass
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        z = ae.encode(images.to(device))[0]
        z = z * scaling_factor
    elif "WanVAE" in name:
        ae = vae

        if isinstance(cache_key, list) and ae.cfg.cache_dir is not None:
            cache_file = [os.path.join(ae.cfg.cache_dir, f"{key}.npz") for key in cache_key]
        else:
            cache_file = None

        z = None
        try:
            if data_info is None:
                z = torch.stack([torch.from_numpy(np.load(cf)["z"]).to(device) for cf in cache_file], dim=0)
            elif data_info is not None and data_info.get("zip_file", None) is not None:
                z = []
                for zip_file, key, dataset_name in zip(
                    data_info["zip_file"], data_info["key"], data_info["dataset_name"]
                ):
                    vae_zip_file = os.path.join(ae.cfg.cache_dir, dataset_name, os.path.basename(zip_file))

                    if os.path.exists(vae_zip_file):
                        from diffusion.data.zip_cache import open_zip_file

                        z_vae_cache = open_zip_file(vae_zip_file)
                        with z_vae_cache.open(key + ".npz", "r") as f:
                            z.append(np.load(f)["z"] if "z" in np.load(f) else np.load(f))
                z = [torch.from_numpy(_z).to(device) for _z in z]
        except:
            z = None

        if z is None or len(z) == 0:
            z = ae.encode(images.to(device))
            if cache_file is not None and if_cache:
                tempdir = os.path.join(ae.cfg.cache_dir, ".tmp")
                os.makedirs(tempdir, exist_ok=True, mode=0o777)
                for i, cf in enumerate(cache_file):
                    if os.path.exists(cf):
                        continue
                    os.makedirs(os.path.dirname(cf), exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=tempdir) as f:
                        np.savez_compressed(f, z=z[i].float().cpu().numpy())  # bf16 not support for cpu
                        try:
                            os.link(f.name, cf)
                        except:
                            pass

        z = torch.stack(z, dim=0)
    elif "Wan2_2_VAE" in name:
        ae = vae
        z = ae.encode(images.to(device))
        z = torch.stack(z, dim=0)
    elif "LTX2VAE_chunk_tile" in name:
        posterior = vae.encode(images.to(device=vae.device, dtype=vae.dtype), causal=True).latent_dist
        z = posterior.mode()
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        z = (z - latents_mean) * vae.config.scaling_factor / latents_std
    elif "LTX2VAE_diffusers" in name:
        # LTX2VAE_diffusers (diffusers AutoencoderKLLTX2Video) expects input shape (B, C, T, H, W) with value range [-1, 1]
        posterior = vae.encode(images.to(device=vae.device, dtype=vae.dtype)).latent_dist
        z = posterior.mode()
        # Normalize latents: z = (z - mean) / std * scaling_factor
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        z = (z - latents_mean) * vae.config.scaling_factor / latents_std
    else:
        print(f"{name} encode error")
        exit()
    return z.to(dtype)


def vae_decode(name, vae, latent):
    """Vae decode.

    Args:
        name: The name.
        vae: The vae.
        latent: The latent.
    """
    if name == "sdxl" or name == "sd3":
        latent = (latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latent).sample
    elif "dc-ae" in name and not "st-dc-ae" in name:
        ae = vae
        vae_scale_factor = (
            2 ** (len(ae.config.encoder_block_out_channels) - 1)
            if hasattr(ae, "config") and ae.config is not None
            else 32
        )
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
            from patch_conv import convert_model

            ae = convert_model(ae, splits=4)
        samples = ae.decode(latent.detach() / scaling_factor)
    elif "dc-vae" in name or "st-dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor is not None else 0.493
        if isinstance(scaling_factor, float):
            latent = latent.detach() / scaling_factor
        else:
            latent = latent.detach() / scaling_factor[None].view(1, -1, 1, 1, 1)
        samples = ae.decode(latent)
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        try:
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
    elif "WanVAE" in name:
        samples = vae.decode(latent)
    elif "Wan2_2_VAE" in name:
        samples = vae.decode(latent)
    elif "LTX2VAE_chunk_tile" in name:
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latent = latent * latents_std / vae.config.scaling_factor + latents_mean
        latent = latent.to(vae.dtype)
        samples = vae.decode_chunk_tile(latent, temb=None, causal=False, return_dict=False)[0]
    elif "LTX2VAE_diffusers" in name:
        # LTX2VAE_diffusers (diffusers AutoencoderKLLTX2Video)
        # Denormalize latents: z = z * std / scaling_factor + mean
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(latent.device, latent.dtype)
        latent = latent * latents_std / vae.config.scaling_factor + latents_mean
        latent = latent.to(vae.dtype)
        # Decode without timestep conditioning (set temb=None)
        samples = vae.decode(latent, temb=None, return_dict=False)[0]
    else:
        print(f"{name} decode error")
        exit()
    return samples


def find_model(model_name: str):
    """Load a Sana checkpoint from a local path or ``hf://`` URI."""
    print(colored(f"[Sana] Loading model from {model_name}", attrs=["bold"]))
    local_path = str(hf_download_or_fpath(model_name))
    assert os.path.isfile(local_path), f"Could not find Sana checkpoint at {local_path}"
    print(colored(f"[Sana] Loaded model from {local_path}", attrs=["bold"]))
    if local_path.endswith((".safetensors", ".safetensors.index.json")):
        return {"state_dict": _load_sana_state_dict(local_path, device="cpu")}
    return require_mapping(
        load_weights_only(local_path, map_location="cpu"),
        source=local_path,
    )
