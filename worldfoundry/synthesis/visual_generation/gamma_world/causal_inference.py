# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import collections
import collections.abc
import os
import time
from typing import Any

import numpy as np
import torch
from einops import rearrange

from worldfoundry.core.distributed import torch_process_group as distributed
from worldfoundry.core.distributed.logging import log
from worldfoundry.core.distributed.megatron_compat import parallel_state
from worldfoundry.core.distributed.model_parallel_state import is_tp_cp_pp_rank0
from worldfoundry.core.io import save_image_or_video_tensor as save_img_or_video
from worldfoundry.core.time import CudaSyncTimer as sync_timer
from worldfoundry.core.utils import inference_runtime as misc
from worldfoundry.synthesis.visual_generation.gamma_world.model_loader import load_model_from_checkpoint

IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"

_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def to_with_skip_tensor(
    data: Any,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    memory_format: torch.memory_format = torch.preserve_format,
    key: str | None = None,
) -> Any:

    skip_tensor_name = [
        "camera",
        "depth",
        "intrinsics",
        "buffer_depths",
        "buffer_w2cs",
        "target_w2cs",
        "buffer_intrinsics",
        "target_intrinsics",
        "buffer_points",
        "buffer_masks",
        "num_video_frames_per_view",
    ]
    assert device is not None or dtype is not None or memory_format is not None, (
        "at least one of device, dtype, memory_format should be specified"
    )
    if isinstance(data, torch.Tensor):
        if (
            memory_format == torch.channels_last
            and data.dim() != 4
            or memory_format == torch.channels_last_3d
            and data.dim() != 5
        ):
            memory_format = torch.preserve_format
        is_cpu = (isinstance(device, str) and device == "cpu") or (
            isinstance(device, torch.device) and device.type == "cpu"
        )
        if not torch.is_floating_point(data):
            data = data.to(
                device=device,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        elif key is not None and key in skip_tensor_name:
            data = data.to(
                device=device,
                dtype=torch.float32,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        else:
            data = data.to(
                device=device,
                dtype=dtype,
                memory_format=memory_format,
                non_blocking=(not is_cpu),
            )
        return data
    elif isinstance(data, collections.abc.Mapping):
        converted = {
            key: to_with_skip_tensor(data[key], device=device, dtype=dtype, memory_format=memory_format, key=key)
            for key in data
        }
        return type(data)(converted)
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, (str, bytes)):
        converted_list = [
            to_with_skip_tensor(elem, device=device, dtype=dtype, memory_format=memory_format, key=key) for elem in data
        ]
        return type(data)(converted_list)
    else:
        return data


def to_model_input(data_batch: dict, model: torch.nn.Module) -> dict:

    for k, v in data_batch.items():
        _v = v
        if isinstance(v, torch.Tensor):
            _v = _v.cuda()
            if torch.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


def save_output(to_show: list[torch.Tensor], vid_save_path: str, fps: int = 16) -> None:

    legancy_to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    video_array = (rearrange(legancy_to_show, "n b c t h w -> t (n h) (b w) c") * 255).to(torch.uint8).cpu().numpy()
    log.info(
        f"video_array.shape: {video_array.shape} value: {video_array.max()}, {video_array.min()}, save to {vid_save_path}"
    )
    save_img_or_video(
        rearrange(legancy_to_show, "n b c t h w -> c t (n h) (b w)"),
        vid_save_path.split(".mp4")[0],
        fps=fps,
    )
    log.info(f"save video to {vid_save_path}", rank0_only=True)


class I2VInference:
    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        config_file: str = "worldfoundry/synthesis/visual_generation/gamma_world/configs/causal/config.py",
        context_parallel_size: int = 1,
        guidance: float = 5.0,
        shift: float = 5.0,
        num_sampling_steps: int = 35,
        seed: int = 1,
        experiment_opts: list[str] | None = None,
        vae_pth: str | None = None,
        text_encoder_pth: str | None = None,
    ):

        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.config_file = config_file
        self.context_parallel_size = context_parallel_size
        self.guidance = guidance
        self.shift = shift
        self.num_sampling_steps = num_sampling_steps
        self.process_group = None

        if "RANK" in os.environ:
            self._init_distributed()

        misc.set_random_seed(seed=seed, by_rank=True)

        if experiment_opts:
            log.info(f"[InferenceI2V] experiment_opts={experiment_opts}")

        self.model, self.config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file=self.config_file,
            cache_text_encoder=True,
            experiment_opts=experiment_opts or [],
            vae_pth=vae_pth,
            text_encoder_pth=text_encoder_pth,
        )

        net_cfg = getattr(self.config.model.config, "net", None)
        if net_cfg is not None:
            log.info(
                "[InferenceI2V] resolved net config: "
                f"use_multi_agent_rope={getattr(net_cfg, 'use_multi_agent_rope', None)} "
                f"num_agents={getattr(net_cfg, 'multi_agent_rope_num_agents', None)} "
                f"simplex_pool_size={getattr(net_cfg, 'multi_agent_rope_simplex_pool_size', None)} "
                f"agent_encoding={getattr(net_cfg, 'multi_agent_rope_agent_encoding', None)} "
                f"agent_scale={getattr(net_cfg, 'multi_agent_rope_agent_scale', None)} "
                f"agent_id_offset={getattr(net_cfg, 'multi_agent_rope_agent_id_offset', None)}"
            )

        self.rank0 = True
        if self.context_parallel_size > 1:
            self.model.net.enable_context_parallel(self.process_group)
            self.rank0 = distributed.get_rank() == 0

        self.model.eval()
        self.model = self.model.to(dtype=torch.bfloat16)

        if hasattr(self.model, "net") and hasattr(self.model.net, "pos_embedder"):
            log.info("Resetting pos_embedder parameters to restore float32 precision after bf16 cast")
            self.model.net.pos_embedder.reset_parameters()
        else:
            log.warning("self.model.net.pos_embedder not available, skipping reset_parameters()")

        self.model.config.split_cp_in_model = False
        self.batch_size = 1
        self.generate_cnt = 0
        torch.cuda.empty_cache()

        if hasattr(self.model, "net") and getattr(self.model.net, "use_multi_agent_rope", False):
            log.info(
                "[multi-agent RoPE] "
                f"use_multi_agent_rope={getattr(self.model.net, 'use_multi_agent_rope', None)} "
                f"num_agents={getattr(self.model.net, 'num_agents', None)} "
                f"simplex_pool_size={getattr(self.model.net, 'simplex_pool_size', None)} "
                f"agent_encoding={getattr(self.model.net, 'agent_encoding', None)} "
                f"agent_scale={getattr(self.model.net, 'agent_scale', None)} "
                f"agent_id_offset={getattr(self.model.net, 'agent_id_offset', None)}"
            )
            if (
                hasattr(self.model.net, "multi_agent_action_control")
                and self.model.net.multi_agent_action_control is not None
            ):
                mac = self.model.net.multi_agent_action_control
                log.info(
                    f"[multi-agent action] type={type(mac).__name__} "
                    f"num_agents={getattr(mac, 'num_agents', None)} "
                    f"pool_size={getattr(mac, 'pool_size', None)}"
                )

    def _init_distributed(self) -> None:

        distributed.init()

        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def clear_cache(self) -> None:

        self.model.kv_cache1 = None
        self.model.kv_cache2 = None

    def build_inference_batch(
        self,
        init_images: list[np.ndarray],
        prompt: str,
        actions: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None,
        *,
        num_frames: int,
        num_conditional_frames: int = 1,
    ) -> dict[str, Any]:

        n_players = len(init_images)
        if actions is not None and len(actions) != n_players:
            raise ValueError(f"len(actions)={len(actions)} != len(init_images)={n_players}")
        height, width = init_images[0].shape[:2]
        per_view = []
        for image in init_images:
            array = np.ascontiguousarray(image)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            chw = torch.from_numpy(array).permute(2, 0, 1)
            video_view = torch.zeros((chw.shape[0], num_frames, height, width), dtype=torch.uint8)
            if num_conditional_frames > 0:
                video_view[:, :num_conditional_frames] = chw.unsqueeze(1)
            per_view.append(video_view)
        video = torch.cat(per_view, dim=1).unsqueeze(0)
        view_indices = torch.tensor(
            [view for view in range(n_players) for _ in range(num_frames)], dtype=torch.int64
        ).unsqueeze(0)
        batch = {
            "video": video,
            self.model.input_caption_key: [[prompt]],
            "view_indices": view_indices,
            "view_indices_selection": torch.arange(n_players, dtype=torch.int64).unsqueeze(0),
            "num_video_frames_per_view": torch.tensor([num_frames], dtype=torch.int64),
            "sample_n_views": torch.tensor([n_players], dtype=torch.int64),
            "fps": torch.tensor([float(getattr(self, "fps", 16.0))], dtype=torch.float64),
            "frame_indices": torch.arange(num_frames, dtype=torch.int64).unsqueeze(0),
            "padding_mask": torch.zeros(1, 1, height, width, dtype=torch.float32),
            "original_hw": torch.tensor([[[height, width]] * n_players], dtype=torch.int64),
            "front_cam_view_idx_sample_position": torch.tensor([0], dtype=torch.int64),
            "ref_cam_view_idx_sample_position": torch.tensor([-1], dtype=torch.int64),
            NUM_CONDITIONAL_FRAMES_KEY: num_conditional_frames,
        }
        for index, (keyboard, camera) in enumerate(actions or []):
            batch[f"action_{index}_keyboard"] = keyboard
            if camera is not None:
                batch[f"action_{index}_camera"] = camera
        return batch

    def inplace_compute_text_embeddings_online(
        self,
        data_batch: dict[str, torch.Tensor],
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> None:

        if (
            self.model.config.text_encoder_config is not None
            and self.model.config.text_encoder_config.compute_online
            and self.model.text_encoder is not None
        ):
            text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                data_batch, self.model.input_caption_key
            )
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

            if use_negative_prompt:
                batch_size = text_embeddings.shape[0]
                neg_data_batch = {self.model.input_caption_key: [negative_prompt] * batch_size, "images": None}
                neg_text_embeddings = self.model.text_encoder.compute_text_embeddings_online(
                    neg_data_batch, self.model.input_caption_key
                )
                data_batch["neg_t5_text_embeddings"] = neg_text_embeddings

    def generate_from_batch(
        self,
        data_batch: dict,
        guidance: float | None = None,
        seed: int = 1,
        num_steps: int | None = None,
        shift: float | None = None,
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
        save_output_for_viz: bool = False,
        output_path: str | None = None,
        output_name: str | None = None,
    ) -> torch.Tensor:

        guidance = guidance if guidance is not None else self.guidance
        num_steps = num_steps if num_steps is not None else self.num_sampling_steps
        shift = shift if shift is not None else self.shift

        if "video" in data_batch:
            data_batch["video"] = data_batch["video"].float()
            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["video"] = data_batch["video"] / 127.5 - 1.0
            data_batch["video"] = torch.clamp(data_batch["video"], -1, 1)

        if "control_input_hdmap_bbox" in data_batch:
            data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"].float()
            if not data_batch.get(IS_PREPROCESSED_KEY, False):
                data_batch["control_input_hdmap_bbox"] = data_batch["control_input_hdmap_bbox"] / 127.5 - 1.0
            data_batch["control_input_hdmap_bbox"] = torch.clamp(data_batch["control_input_hdmap_bbox"], -1, 1)

        data_batch[IS_PREPROCESSED_KEY] = True
        data_batch = to_with_skip_tensor(data_batch, **self.model.tensor_kwargs)

        self.inplace_compute_text_embeddings_online(
            data_batch,
            use_negative_prompt=use_negative_prompt,
            negative_prompt=negative_prompt,
        )

        if hasattr(self.model, "net") and hasattr(self.model.net, "multi_agent_action_control"):
            mac = self.model.net.multi_agent_action_control
            if mac is not None and "action_inference_scale" in data_batch:
                override_val = data_batch.pop("action_inference_scale")
                log.info(f"Overriding action scale: {mac.scale.item():.6f} -> {override_val}")
                mac.scale.data.fill_(override_val)

        control_input_hdmap_bbox_viz = data_batch.get("control_input_hdmap_bbox")
        data_batch = self.model.get_data_batch_with_latent_view_indices(data_batch)
        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)

        with torch.no_grad():
            log.info("Start inference", rank0_only=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            _gen_t0 = time.perf_counter()
            with sync_timer("generate_samples_from_batch"):
                sample = self.model.generate_samples_from_batch(
                    data_batch,
                    guidance=guidance,
                    shift=shift,
                    state_shape=x0.shape[1:],
                    n_sample=x0.shape[0],
                    seed=seed,
                    num_steps=num_steps,
                    is_negative_prompt=use_negative_prompt,
                    verbose=True,
                )
            torch.cuda.synchronize()
            _sample_elapsed = time.perf_counter() - _gen_t0
            _sample_peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            _sample_peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            torch.cuda.reset_peak_memory_stats()
            _dec_t0 = time.perf_counter()
            with sync_timer("decode"):
                video = self.model.decode(sample)
            torch.cuda.synchronize()
            _decode_elapsed = time.perf_counter() - _dec_t0
            _decode_peak_alloc = torch.cuda.max_memory_allocated() / (1024**3)
            _decode_peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)

            log.info(
                f"[gen-stats] sample: time={_sample_elapsed:.2f}s "
                f"peak_alloc={_sample_peak_alloc:.2f}GB peak_reserved={_sample_peak_reserved:.2f}GB | "
                f"decode: time={_decode_elapsed:.2f}s "
                f"peak_alloc={_decode_peak_alloc:.2f}GB peak_reserved={_decode_peak_reserved:.2f}GB | "
                f"total_gen_time={_sample_elapsed + _decode_elapsed:.2f}s",
                rank0_only=True,
            )
            log.info("End inference", rank0_only=True)

        n_views = int(data_batch["sample_n_views"].cpu().item())
        if n_views > 1:
            video = rearrange(video, "B C (V T) H W -> B C T H (V W)", V=n_views)

        if save_output_for_viz and output_path is not None:
            os.makedirs(output_path, exist_ok=True)

            if output_name is not None:
                base_fp_wo_ext = os.path.join(output_path, output_name + "_with_hdmap.mp4")
            else:
                base_fp_wo_ext = os.path.join(output_path, f"_Sample_Iter{self.generate_cnt:03d}.mp4")
            self.generate_cnt += 1
            to_show = [
                video.float().cpu(),
            ]
            if control_input_hdmap_bbox_viz is not None:
                to_show.insert(0, control_input_hdmap_bbox_viz.float().cpu())
            if self.context_parallel_size > 1:
                if is_tp_cp_pp_rank0():
                    save_output(to_show, base_fp_wo_ext)
            else:
                save_output(to_show, base_fp_wo_ext)

        return video

    def cleanup(self) -> None:

        if "RANK" in os.environ:
            import torch.distributed as dist

            from worldfoundry.core.distributed.megatron_compat import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
