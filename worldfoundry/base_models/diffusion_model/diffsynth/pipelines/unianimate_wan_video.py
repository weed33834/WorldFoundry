"""UniAnimate Wan inference pipeline for the in-tree DiffSynth runtime."""

import os
import types

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from ..models import ModelManager
from ..schedulers.flow_match_pusa import FlowMatchSchedulerPusa
from .wan_video import TeaCache, WanVideoPipeline, model_fn_wan_video
from .wan_video_pusa import TeaCache as PusaTeaCache
from .wan_video_pusa import model_fn_wan_video as model_fn_wan_video_pusa


class WanUniAnimateVideoPipeline(WanVideoPipeline):
    """Wan2.1 UniAnimate inference pipeline.

    This class keeps the UniAnimate-specific pose conditioning path in-tree
    while reusing the WorldFoundry Wan video pipeline base implementation.
    """

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype, tokenizer_path=tokenizer_path)
        self.model_names = ["text_encoder", "dit", "vae"]
        self.use_unified_sequence_parallel = False
        self.dwpose_embedding = None
        self.randomref_embedding_pose = None
        self._uses_pusa_dit = False

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self._uses_pusa_dit = False
        if self.dit is None:
            self.dit = model_manager.fetch_model("wan_video_pusa")
            self._uses_pusa_dit = self.dit is not None
            if self._uses_pusa_dit:
                self.scheduler = FlowMatchSchedulerPusa(shift=5, sigma_min=0.0, extra_one_step=True)
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

        concat_dim = 4
        self.dwpose_embedding = nn.Sequential(
            nn.Conv3d(3, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, concat_dim * 4, 3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(concat_dim * 4, 5120, (1, 2, 2), stride=(1, 2, 2), padding=0),
        )

        randomref_dim = 20
        self.randomref_embedding_pose = nn.Sequential(
            nn.Conv2d(3, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, concat_dim * 4, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(concat_dim * 4, randomref_dim, 3, stride=2, padding=1),
        )

        dwpose_state = {
            key.split("dwpose_embedding.")[1]: value
            for key, value in model_manager.state_dict_new_module.items()
            if "dwpose_embedding" in key
        }
        self.dwpose_embedding.load_state_dict(dwpose_state, strict=True)

        randomref_state = {
            key.split("randomref_embedding_pose.")[1]: value
            for key, value in model_manager.state_dict_new_module.items()
            if "randomref_embedding_pose" in key
        }
        self.randomref_embedding_pose.load_state_dict(randomref_state, strict=True)

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None, use_usp=False):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanUniAnimateVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size
            from worldfoundry.core.attention.patch_xdit_context_parallel import usp_attn_forward, usp_dit_forward

            for block in pipe.dit.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            pipe.dit.forward = types.MethodType(usp_dit_forward, pipe.dit)
            pipe.sp_size = get_sequence_parallel_world_size()
            pipe.use_unified_sequence_parallel = True

        return pipe

    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
        msk = msk.transpose(1, 2)[0]

        vae_input = torch.concat(
            [image.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(image.device)],
            dim=1,
        )
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        dwpose_data=None,
        random_ref_dwpose=None,
    ):
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 == 1` is acceptable. We round it up to {num_frames}.")
        if dwpose_data is None or random_ref_dwpose is None:
            raise ValueError("UniAnimate requires dwpose_data and random_ref_dwpose inputs.")

        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)

        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        )
        noise = noise.to(dtype=self.torch_dtype, device=self.device)
        if input_video is not None:
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2).to(dtype=self.torch_dtype, device=self.device)
            latents = self.encode_video(input_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)
            latents = self.scheduler.add_noise(latents, noise, timestep=self.scheduler.timesteps[0])
        else:
            latents = noise

        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {}

        extra_input = self.prepare_extra_input(latents)
        tea_cache_cls = PusaTeaCache if self._uses_pusa_dit else TeaCache
        tea_cache_posi = {
            "tea_cache": tea_cache_cls(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
            if tea_cache_l1_thresh is not None
            else None
        }
        tea_cache_nega = {
            "tea_cache": tea_cache_cls(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)
            if tea_cache_l1_thresh is not None
            else None
        }

        self.load_models_to_device(["dit"])
        usp_kwargs = self.prepare_unified_sequence_parallel()

        self.dwpose_embedding.to(self.device)
        self.randomref_embedding_pose.to(self.device)
        dwpose_data = dwpose_data.unsqueeze(0)
        dwpose_data = self.dwpose_embedding(
            (torch.cat([dwpose_data[:, :, :1].repeat(1, 1, 3, 1, 1), dwpose_data], dim=2) / 255.0).to(self.device)
        ).to(torch.bfloat16)
        random_ref_dwpose_data = self.randomref_embedding_pose(
            (random_ref_dwpose.unsqueeze(0) / 255.0).to(self.device).permute(0, 3, 1, 2)
        ).unsqueeze(2).to(torch.bfloat16)

        image_emb["y"] = image_emb["y"] + random_ref_dwpose_data
        condition = rearrange(dwpose_data, "b c f h w -> b (f h w) c").contiguous()
        denoise_fn = model_fn_wan_video_pusa if self._uses_pusa_dit else model_fn_wan_video
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            if self._uses_pusa_dit:
                model_timestep = (
                    timestep.unsqueeze(0)
                    .unsqueeze(1)
                    .repeat(1, latents.shape[2])
                    .to(dtype=self.torch_dtype, device=self.device)
                )
                scheduler_timestep = model_timestep
            else:
                model_timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
                scheduler_timestep = self.scheduler.timesteps[progress_id]

            noise_pred_posi = denoise_fn(
                self.dit,
                x=latents,
                timestep=model_timestep,
                **prompt_emb_posi,
                **image_emb,
                **extra_input,
                **tea_cache_posi,
                **usp_kwargs,
                add_condition=condition,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = denoise_fn(
                    self.dit,
                    x=latents,
                    timestep=model_timestep,
                    **prompt_emb_nega,
                    **image_emb,
                    **extra_input,
                    **tea_cache_nega,
                    **usp_kwargs,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = self.scheduler.step(noise_pred, scheduler_timestep, latents)

        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        return self.tensor2video(frames[0])
