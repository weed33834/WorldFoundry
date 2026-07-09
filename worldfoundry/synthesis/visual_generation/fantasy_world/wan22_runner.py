from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import random
import sys
import torch
from PIL import Image
from tqdm import tqdm

from .runtime_env import (
    ensure_fantasy_world_runtime,
    ensure_moge2_runtime,
    prepare_wan22_runtime_root,
    resolve_moge_pretrained,
)
from .worldfoundry_runtime import normalize_wan_num_frames, pad_camera_params_to_frames


def _checkpoint_uses_control_adapter(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("pipe.dit.control_adapter.") for key in state_dict)


def _ensure_wan22_control_adapter(model) -> None:
    dit = model.pipe.dit
    if getattr(dit, "control_adapter", None) is not None:
        return
    from worldfoundry.base_models.diffusion_model.diffsynth.models.fantasy_world_wan22_wan_video_dit import (
        SimpleAdapter,
    )

    reference = next(dit.parameters())
    dit.control_adapter = SimpleAdapter(
        24,
        dit.dim,
        kernel_size=dit.patch_size[1:],
        stride=dit.patch_size[1:],
    ).to(device=reference.device, dtype=reference.dtype)


class FantasyWorldWan22Runner:
    """Official FantasyWorld Wan2.2 inference wrapper."""

    @staticmethod
    def _canonical_cuda_device(device: str) -> str:
        if device == "cuda":
            return "cuda:0"
        if device.startswith("cuda:"):
            return device
        return device

    @classmethod
    def _auto_device_layout(
        cls,
        base_device: str,
        high_model_device: Optional[str],
        low_model_device: Optional[str],
        moge_device: Optional[str],
    ) -> tuple[str, str, str]:
        resolved_base = cls._canonical_cuda_device(base_device)
        if not resolved_base.startswith("cuda"):
            return resolved_base, resolved_base, resolved_base

        device_count = torch.cuda.device_count()
        if device_count <= 1:
            resolved_high = cls._canonical_cuda_device(high_model_device or resolved_base)
            resolved_low = cls._canonical_cuda_device(low_model_device or resolved_high)
            resolved_moge = cls._canonical_cuda_device(moge_device or resolved_high)
            return resolved_high, resolved_low, resolved_moge

        base_index = 0 if resolved_base == "cuda" else int(resolved_base.split(":", maxsplit=1)[1])
        resolved_high = cls._canonical_cuda_device(high_model_device or f"cuda:{base_index}")
        high_index = int(resolved_high.split(":", maxsplit=1)[1])

        if low_model_device is None:
            resolved_low = f"cuda:{(high_index + 1) % device_count}"
        else:
            resolved_low = cls._canonical_cuda_device(low_model_device)

        if moge_device is None:
            candidate_indices = [
                idx for idx in range(device_count)
                if f"cuda:{idx}" not in {resolved_high, resolved_low}
            ]
            resolved_moge = f"cuda:{candidate_indices[0]}" if candidate_indices else resolved_high
        else:
            resolved_moge = cls._canonical_cuda_device(moge_device)

        return resolved_high, resolved_low, resolved_moge

    def __init__(
        self,
        *,
        ckpt_dir: str,
        model_ckpt_high: str,
        model_ckpt_low: str,
        moge_path: Optional[str] = None,
        moge_pretrained: Optional[str] = None,
        base_seed: int = -1,
        sample_steps: int = 50,
        cfg_scale: float = 5.0,
        timestep_boundary: int = 900,
        frames: int = 81,
        fps: int = 16,
        height: int = 480,
        width: int = 832,
        device: str = "cuda",
        high_model_device: Optional[str] = None,
        low_model_device: Optional[str] = None,
        moge_device: Optional[str] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("FantasyWorld Wan2.2 official inference requires a CUDA device.")

        ensure_fantasy_world_runtime()
        ensure_moge2_runtime(moge_path)

        from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel
        from worldfoundry.base_models.diffusion_model.diffsynth.utils.re10k_pose import (
            RealEstate10KPoseProcessor,
        )
        from FantasyWorld.fusion.model_wan22 import FantasyWorldFusionModel
        from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.variants.fantasy_world.utils.pose_enc import (
            extri_intri_to_pose_encoding,
            pose_encoding_to_extri_intri,
        )

        from . import utils as fw_utils

        self.base_seed = int(base_seed) if int(base_seed) >= 0 else random.randint(0, sys.maxsize)
        self.sample_steps = int(sample_steps)
        self.cfg_scale = float(cfg_scale)
        self.fps = int(fps)
        self.high_device, self.low_device, self.moge_device = self._auto_device_layout(
            str(device),
            high_model_device,
            low_model_device,
            moge_device,
        )
        self.device = self.high_device
        self.torch_dtype = weight_dtype
        self.num_frames = normalize_wan_num_frames(frames)
        self.height = int(height)
        self.width = int(width)
        self.timestep_boundary = int(timestep_boundary)

        self._fw_utils = fw_utils
        self._extri_intri_to_pose_encoding = extri_intri_to_pose_encoding

        vggt_cfg = {
            "enable_camera": True,
            "enable_depth": True,
            "enable_point": True,
            "enable_track": False,
            "DPT_patch_size": 16,
        }
        camera_cfg = {
            "pose_in_dim": 768,
            "plucker_fea_dim": 2048,
            "pose_inject_method": "adaln",
            "use_info": "plucker",
        }

        self.model_high = FantasyWorldFusionModel(
            start_index=16,
            use_gradient_checkpointing=True,
            cross_attention_list=list(range(24)),
            origin_file_pattern="high_noise_model/diffusion_pytorch_model*.safetensors",
            dit_path=ckpt_dir,
            lora_path=f"{ckpt_dir}/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-high-noise-HPS2.1.safetensors",
            vggt_cfg=vggt_cfg,
            camera_control=True,
            camera_cfg=camera_cfg,
            load_vae=True,
            load_text_encoder=True,
        )
        self.model_low = FantasyWorldFusionModel(
            start_index=16,
            use_gradient_checkpointing=True,
            cross_attention_list=list(range(24)),
            origin_file_pattern="low_noise_model/diffusion_pytorch_model*.safetensors",
            dit_path=ckpt_dir,
            lora_path=f"{ckpt_dir}/PAI/Wan2.2-Fun-Reward-LoRAs/Wan2.2-Fun-A14B-InP-low-noise-HPS2.1.safetensors",
            vggt_cfg=vggt_cfg,
            camera_control=True,
            camera_cfg=camera_cfg,
        )

        ckpt_high = torch.load(model_ckpt_high, map_location="cpu")
        if _checkpoint_uses_control_adapter(ckpt_high):
            _ensure_wan22_control_adapter(self.model_high)
        messages = self.model_high.load_state_dict(ckpt_high, strict=False)
        if messages.unexpected_keys:
            raise RuntimeError(f"Unexpected FantasyWorld Wan2.2 HIGH keys: {messages.unexpected_keys}")
        del ckpt_high

        ckpt_low = torch.load(model_ckpt_low, map_location="cpu")
        if _checkpoint_uses_control_adapter(ckpt_low):
            _ensure_wan22_control_adapter(self.model_low)
        messages = self.model_low.load_state_dict(ckpt_low, strict=False)
        if messages.unexpected_keys:
            raise RuntimeError(f"Unexpected FantasyWorld Wan2.2 LOW keys: {messages.unexpected_keys}")
        del ckpt_low

        self.model_high.to(self.torch_dtype)
        self.model_high.to(self.high_device)
        self.model_high.pipe.device = self.high_device
        self.model_high.eval()

        self.model_low.to(self.torch_dtype)
        self.model_low.to(self.low_device)
        self.model_low.pipe.device = self.low_device
        self.model_low.eval()

        self.pose_processor = RealEstate10KPoseProcessor(
            sample_stride=1,
            sample_n_frames=self.num_frames,
            relative_pose=True,
            zero_t_first_frame=True,
            sample_size=[self.height, self.width],
            rescale_fxy=False,
            shuffle_frames=False,
            use_flip=False,
            is_i2v=True,
            pose_encoding_to_extri_intri=pose_encoding_to_extri_intri,
        )
        self.moge = MoGeModel.from_pretrained(resolve_moge_pretrained(moge_pretrained)).to(self.moge_device).eval()

    def generate_video_with_dual_models(
        self,
        *,
        context_pos: torch.Tensor,
        context_neg: torch.Tensor,
        y: torch.Tensor,
        plucker_embedding: torch.Tensor,
    ):
        num_frames = self.num_frames
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1

        self.model_high.pipe.scheduler.set_timesteps(self.sample_steps)
        self.model_low.pipe.scheduler.set_timesteps(self.sample_steps)

        noise = self.model_high.pipe.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, self.height // 8, self.width // 8),
            seed=self.base_seed,
        ).to(dtype=self.torch_dtype, device=self.high_device)
        latents = noise

        def _prepare_control_latents(target_device: str) -> torch.Tensor:
            control_camera_video = plucker_embedding.to(
                device=target_device,
                dtype=self.torch_dtype,
            )[0].permute([3, 0, 1, 2]).unsqueeze(0)
            control_camera_latents = torch.concat(
                [
                    torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                    control_camera_video[:, :, 1:],
                ],
                dim=2,
            ).transpose(1, 2)
            bsz, frames_local, channels, height, width = control_camera_latents.shape
            control_camera_latents = control_camera_latents.contiguous().view(
                bsz, frames_local // 4, 4, channels, height, width
            ).transpose(2, 3)
            control_camera_latents = control_camera_latents.contiguous().view(
                bsz, frames_local // 4, channels * 4, height, width
            ).transpose(1, 2)
            return control_camera_latents.to(device=target_device, dtype=self.torch_dtype)

        control_latents_by_device = {
            self.high_device: _prepare_control_latents(self.high_device),
        }
        if self.low_device != self.high_device:
            control_latents_by_device[self.low_device] = _prepare_control_latents(self.low_device)

        y_by_device = {self.high_device: y.to(dtype=self.torch_dtype, device=self.high_device)}
        if self.low_device != self.high_device:
            y_by_device[self.low_device] = y.to(dtype=self.torch_dtype, device=self.low_device)

        context_pos_by_device = {
            self.high_device: context_pos.to(dtype=self.torch_dtype, device=self.high_device),
        }
        context_neg_by_device = {
            self.high_device: context_neg.to(dtype=self.torch_dtype, device=self.high_device),
        }
        if self.low_device != self.high_device:
            context_pos_by_device[self.low_device] = context_pos.to(
                dtype=self.torch_dtype,
                device=self.low_device,
            )
            context_neg_by_device[self.low_device] = context_neg.to(
                dtype=self.torch_dtype,
                device=self.low_device,
            )
        final_prediction = None

        for progress_id, _ in enumerate(tqdm(range(self.sample_steps))):
            step_t = self.model_high.pipe.scheduler.timesteps[progress_id]
            current_model = self.model_high if step_t.item() > self.timestep_boundary else self.model_low
            current_device = self.high_device if current_model is self.model_high else self.low_device
            t = step_t.unsqueeze(0).to(dtype=self.torch_dtype, device=current_device)
            latents = latents.to(current_device)

            noise_pred_posi, prediction = current_model.joint_forward(
                latents,
                timestep=t,
                context=context_pos_by_device[current_device],
                y=y_by_device[current_device],
                use_gradient_checkpointing=False,
                camera_token=None,
                control_camera_latents_input=control_latents_by_device[current_device],
                uncond=False,
                return_prediction=progress_id == self.sample_steps - 1,
            )

            if self.cfg_scale != 1.0 and context_neg is not None:
                noise_pred_nega, _ = current_model.joint_forward(
                    latents,
                    timestep=t,
                    context=context_neg_by_device[current_device],
                    y=y_by_device[current_device],
                    use_gradient_checkpointing=False,
                    camera_token=None,
                    control_camera_latents_input=control_latents_by_device[current_device],
                    uncond=False,
                )
                noise_pred = noise_pred_nega + self.cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            latents = current_model.pipe.scheduler.step(
                noise_pred,
                current_model.pipe.scheduler.timesteps[progress_id],
                latents,
            )
            final_prediction = prediction

        return latents, final_prediction

    def generate_video(
        self,
        *,
        image: Image.Image,
        end_image: Optional[Image.Image],
        prompt: str,
        neg_prompt: Optional[str],
        camera_params,
        using_scale: bool = True,
    ):
        neg_prompt = neg_prompt or ""

        with torch.no_grad():
            input_image = image.convert("RGB")
            camera_params = pad_camera_params_to_frames(camera_params, self.num_frames)
            input_image_pt = torch.tensor(
                np.array(input_image) / 255,
                dtype=torch.float32,
                device=self.moge_device,
            ).permute(2, 0, 1)

            output = self.moge.infer(input_image_pt)
            moge = {k: v.cpu().contiguous() for k, v in output.items()}

            intrinsics = []
            extrinsics = []
            for camera in camera_params:
                intrinsics.append(self._fw_utils.get_intrinsic_matrix(camera))
                extrinsics.append(camera.w2c_mat)
            intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))
            extrinsics = torch.from_numpy(np.stack(extrinsics).astype(np.float32))
            extrinsics_4x4 = extrinsics.unsqueeze(0)

            if using_scale:
                first_intrinsic = intrinsics[0, :, :].unsqueeze(0)
                first_extrinsic = extrinsics[0, :3, :].unsqueeze(0)
                first_moge_world, first_moge_mask = self._fw_utils.batch_depth_to_world(
                    prediction=moge,
                    extrinsics=first_extrinsic,
                    intrinsics=first_intrinsic,
                )
                extrinsics_3x4 = extrinsics_4x4[:, :, :3, :]
                extrinsics = self._fw_utils.normalize_scene(
                    extrinsics=extrinsics_3x4,
                    first_moge_world=first_moge_world.unsqueeze(0),
                    first_moge_mask=first_moge_mask.unsqueeze(0),
                ).squeeze(0)

            pose_enc = self._extri_intri_to_pose_encoding(
                extrinsics.unsqueeze(0),
                intrinsics.unsqueeze(0),
                [self.height, self.width],
                pose_encoding_type="absT_quaR_FoV",
            ).squeeze(0)
            plucker_embedding = self.pose_processor.get_plucker_embedding_direct_from_cam_params(
                pose_enc.unsqueeze(0),
                image_size=(self.height, self.width),
            ).to(dtype=self.torch_dtype)

            inputs_shared, inputs_posi, inputs_nega = self.model_high.pipe(
                prompt=prompt or "",
                negative_prompt=neg_prompt,
                seed=self.base_seed,
                tiled=True,
                input_image=input_image,
                end_image=end_image,
                height=self.height,
                width=self.width,
                num_frames=self.num_frames,
                return_condition=True,
            )
            ctx_pos, ctx_neg = inputs_posi["context"], inputs_nega["context"]
            y = inputs_shared["y"]

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            latent_video, prediction = self.generate_video_with_dual_models(
                context_pos=ctx_pos,
                context_neg=ctx_neg,
                y=y,
                plucker_embedding=plucker_embedding,
            )
            latent_video = latent_video.to(self.high_device)
            frames = self.model_high.pipe.vae.decode(
                latent_video,
                device=self.high_device,
                tiled=True,
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )

        video = frames.squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu()
        video = (video + 1.0) / 2.0
        video = (video * 255.0).clamp(0, 255)
        frames_np_processed = video.numpy().astype(np.uint8)
        return frames_np_processed, prediction


def build_wan22_runner(
    *,
    base_dir: str,
    lora_dir: str,
    model_ckpt_high: str,
    model_ckpt_low: str,
    moge_path: Optional[str] = None,
    moge_pretrained: Optional[str] = None,
    base_seed: int = -1,
    sample_steps: int = 50,
    cfg_scale: float = 5.0,
    timestep_boundary: int = 900,
    frames: int = 81,
    fps: int = 16,
    height: int = 480,
    width: int = 832,
    device: str = "cuda",
    high_model_device: Optional[str] = None,
    low_model_device: Optional[str] = None,
    moge_device: Optional[str] = None,
    weight_dtype: torch.dtype = torch.bfloat16,
):
    runtime_root, runtime_handle = prepare_wan22_runtime_root(base_dir, lora_dir)
    runner = FantasyWorldWan22Runner(
        ckpt_dir=runtime_root,
        model_ckpt_high=model_ckpt_high,
        model_ckpt_low=model_ckpt_low,
        moge_path=moge_path,
        moge_pretrained=moge_pretrained,
        base_seed=base_seed,
        sample_steps=sample_steps,
        cfg_scale=cfg_scale,
        timestep_boundary=timestep_boundary,
        frames=frames,
        fps=fps,
        height=height,
        width=width,
        device=device,
        high_model_device=high_model_device,
        low_model_device=low_model_device,
        moge_device=moge_device,
        weight_dtype=weight_dtype,
    )
    runner._runtime_handle = runtime_handle
    return runner
