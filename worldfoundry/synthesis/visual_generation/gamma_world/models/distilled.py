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

import copy
from typing import Dict, List, Literal, Optional, Tuple

import attrs
import torch
import torch.distributed as dist
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask

from worldfoundry.core.distributed.context_parallel import (
    broadcast,
    broadcast_split_tensor,
    cat_outputs_cp_with_grad,
)
from worldfoundry.core.distributed.megatron_compat import parallel_state
from worldfoundry.core.nn.diffusion_schedulers import FlowMatchScheduler
from worldfoundry.core.utils import inference_runtime as misc
from worldfoundry.synthesis.visual_generation.gamma_world.conditioning.base import DataType
from worldfoundry.synthesis.visual_generation.gamma_world.models.distilled_base import (
    DMDInferenceBaseModel,
    DMDInferenceBaseModelConfig,
)


@attrs.define(slots=False)
class BaseModelConfig(DMDInferenceBaseModelConfig):
    denoising_step_list: List[int] = [1000, 750, 500, 250]
    warp_denoising_step: bool = True
    num_diffusion_timesteps: int = 1000


class BaseModel(DMDInferenceBaseModel):
    config: BaseModelConfig

    def __init__(self, config: BaseModelConfig):
        super().__init__(config)
        self.scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(config.num_diffusion_timesteps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

        self.denoising_step_list = torch.LongTensor(config.denoising_step_list)

        if self.config.warp_denoising_step:
            timesteps = torch.cat(
                (
                    self.scheduler.timesteps.cpu(),
                    torch.tensor([0], dtype=torch.float32),
                )
            )
            self.denoising_step_list = timesteps[self.config.num_diffusion_timesteps - self.denoising_step_list]


@attrs.define(slots=False)
class SelfForcingModelConfig(BaseModelConfig):
    num_cache_frames: int = 21
    num_frame_per_block: int = 3
    context_noise: float = 0.0

    model_type: str = "t2v"
    i2v_zero_latent_condition: bool = False

    min_num_conditional_frames: int = 0
    max_num_conditional_frames: int = 1
    conditional_frame_timestep: float = -1.0
    conditioning_strategy: str = "frame_replace"
    denoise_replace_gt_frames: bool = False
    conditional_frames_probs: Optional[Dict[int, float]] = None


class SelfForcingModel(BaseModel):
    config: SelfForcingModelConfig

    def __init__(self, config: SelfForcingModelConfig):
        super().__init__(config)
        if hasattr(self.net, "num_frame_per_block"):
            self.net.num_frame_per_block = config.num_frame_per_block

    def generator(self, *args, **kwargs):
        return self.denoise(net_choice="generator", scheduler=self.scheduler, *args, **kwargs)


@attrs.define(slots=False)
class DMDSelfForcingModelConfig(SelfForcingModelConfig):
    image_or_video_shape: list[int] = [16, 21, 60, 104]
    state_ch: int = 16
    state_t: int = 24

    noise_scheme: str = "diffusion_forcing"


NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


class DMDSelfForcingModel(SelfForcingModel):
    config: DMDSelfForcingModelConfig

    def __init__(self, config: DMDSelfForcingModelConfig):

        super().__init__(config)

        self.frame_seq_length = None

    def _extract_extra_conditional_kwargs(self, data_batch: dict[str, torch.Tensor]) -> dict:

        return {}

    def denoise(
        self,
        scheduler,
        net_choice: Literal["generator", "real_score", "fake_score"],
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None,
        start_frame_for_rope: Optional[int] = None,
        block_mask: Optional[BlockMask] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if net_choice != "generator":
            raise ValueError("the inference-only model exposes only the generator")
        model = self.net

        n_views = noisy_image_or_video.shape[1] // self.config.state_t

        xt_B_C_T_H_W = noisy_image_or_video.permute(0, 2, 1, 3, 4)
        timesteps_B_T = timestep

        condition_video_mask = None

        if True:
            condition_state_in_B_C_T_H_W = conditional_dict["gt_frames"].type_as(xt_B_C_T_H_W)
            if not conditional_dict["use_video_condition"]:
                condition_state_in_B_C_T_H_W = condition_state_in_B_C_T_H_W * 0

            _, C, _, _, _ = xt_B_C_T_H_W.shape
            condition_video_mask = (
                conditional_dict["condition_video_input_mask_B_C_T_H_W"].repeat(1, C, 1, 1, 1).type_as(xt_B_C_T_H_W)
            )

            xt_B_C_T_H_W = condition_state_in_B_C_T_H_W * condition_video_mask + xt_B_C_T_H_W * (
                1 - condition_video_mask
            )

            if self.config.conditional_frame_timestep >= 0:
                condition_video_mask_B_1_T_1_1 = condition_video_mask.mean(dim=[1, 3, 4], keepdim=True)
                timestep_cond_B_1_T_1_1 = (
                    torch.ones_like(condition_video_mask_B_1_T_1_1) * self.config.conditional_frame_timestep
                )

                timesteps_B_1_T_1_1 = timestep_cond_B_1_T_1_1 * condition_video_mask_B_1_T_1_1 + timesteps_B_T * (
                    1 - condition_video_mask_B_1_T_1_1
                )

                timesteps_B_T = timesteps_B_1_T_1_1.squeeze()
                timesteps_B_T = timesteps_B_T.unsqueeze(0) if timesteps_B_T.ndim == 1 else timesteps_B_T

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1 and hasattr(model, "enable_context_parallel"):
            model.enable_context_parallel(cp_group)
        if cp_size == 1 and hasattr(model, "disable_context_parallel"):
            model.disable_context_parallel()

        if net_choice in ["real_score", "fake_score"]:
            if cp_size > 1 and n_views == 1:
                input_xt_B_C_T_H_W = broadcast_split_tensor(xt_B_C_T_H_W, seq_dim=2, process_group=cp_group)
                if timesteps_B_T.shape[1] > 1:
                    input_timesteps_B_T = broadcast_split_tensor(timesteps_B_T, seq_dim=1, process_group=cp_group)
                else:
                    input_timesteps_B_T = timesteps_B_T

                gt_frames = conditional_dict.get("gt_frames")
                cond_mask = conditional_dict.get("condition_video_input_mask_B_C_T_H_W")
                view_indices = conditional_dict.get("view_indices_B_T")
                control_input_hdmap_bbox = conditional_dict.get("control_input_hdmap_bbox")
                state_t = self.config.state_t

                input_conditional_dict = {}
                for k, v in conditional_dict.items():
                    if k in {"gt_frames", "condition_video_input_mask_B_C_T_H_W", "view_indices_B_T"}:
                        continue
                    if v is None:
                        input_conditional_dict[k] = None
                    elif not isinstance(v, torch.Tensor):
                        input_conditional_dict[k] = v
                    else:
                        input_conditional_dict[k] = broadcast(v, cp_group)

                if gt_frames is not None and cond_mask is not None and view_indices is not None:
                    _, _, T, _, _ = gt_frames.shape
                    assert T % state_t == 0, f"T must be a multiple of state_t. Got T={T} and state_t={state_t}."
                    if T > 1 and cp_group.size() > 1:
                        n_views = T // state_t
                        gt_frames = rearrange(gt_frames, "B C (V T) H W -> B C V T H W", V=n_views)
                        cond_mask = rearrange(cond_mask, "B C (V T) H W -> B C V T H W", V=n_views)
                        view_indices = rearrange(view_indices, "B (V T) -> B V T", V=n_views)

                        gt_frames = broadcast_split_tensor(gt_frames, seq_dim=3, process_group=cp_group)
                        cond_mask = broadcast_split_tensor(cond_mask, seq_dim=3, process_group=cp_group)
                        view_indices = broadcast_split_tensor(view_indices, seq_dim=2, process_group=cp_group)

                        gt_frames = rearrange(gt_frames, "B C V T H W -> B C (V T) H W", V=n_views)
                        cond_mask = rearrange(cond_mask, "B C V T H W -> B C (V T) H W", V=n_views)
                        view_indices = rearrange(view_indices, "B V T -> B (V T)", V=n_views)
                        if control_input_hdmap_bbox is not None:
                            control_input_hdmap_bbox_B_C_V_T_H_W = rearrange(
                                control_input_hdmap_bbox, "B C (V T) H W -> B C V T H W", V=n_views
                            )
                            control_input_hdmap_bbox_B_C_V_T_H_W = broadcast_split_tensor(
                                control_input_hdmap_bbox_B_C_V_T_H_W, seq_dim=3, process_group=cp_group
                            )
                            control_input_hdmap_bbox = rearrange(
                                control_input_hdmap_bbox_B_C_V_T_H_W, "B C V T H W -> B C (V T) H W", V=n_views
                            )
                    else:
                        gt_frames = broadcast(gt_frames, cp_group)
                        cond_mask = broadcast(cond_mask, cp_group)
                        view_indices = broadcast(view_indices, cp_group)
                        if control_input_hdmap_bbox is not None:
                            control_input_hdmap_bbox = broadcast(control_input_hdmap_bbox, cp_group)

                input_conditional_dict["gt_frames"] = gt_frames
                input_conditional_dict["condition_video_input_mask_B_C_T_H_W"] = cond_mask
                input_conditional_dict["view_indices_B_T"] = view_indices
                input_conditional_dict["control_input_hdmap_bbox"] = control_input_hdmap_bbox
            else:
                input_xt_B_C_T_H_W, input_conditional_dict, input_timesteps_B_T = (
                    xt_B_C_T_H_W,
                    conditional_dict,
                    timesteps_B_T,
                )

            flow_pred = model(
                input_xt_B_C_T_H_W.to(**self.tensor_kwargs),
                input_timesteps_B_T.to(**self.tensor_kwargs),
                block_mask=block_mask,
                **input_conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

            if cp_size > 1 and n_views == 1:
                flow_pred = cat_outputs_cp_with_grad(flow_pred.contiguous(), seq_dim=1, cp_group=cp_group)

        else:
            assert net_choice == "generator"
            assert kv_cache is not None
            flow_pred = model(
                xt_B_C_T_H_W.to(**self.tensor_kwargs),
                timesteps_B_T.to(**self.tensor_kwargs),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
                block_mask=block_mask,
                **conditional_dict,
                **kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            scheduler=scheduler,
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        if self.config.denoise_replace_gt_frames:
            gt_frames_x0 = conditional_dict["gt_frames"].type_as(pred_x0)
            pred_x0 = (
                gt_frames_x0 * condition_video_mask + pred_x0.permute(0, 2, 1, 3, 4) * (1 - condition_video_mask)
            ).permute(0, 2, 1, 3, 4)

        return flow_pred, pred_x0

    def get_data_batch_with_latent_view_indices(self, data_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "latent_view_indices_B_T" in data_batch:
            return data_batch
        num_video_frames_per_view = int(data_batch["num_video_frames_per_view"].cpu().item())
        n_views = data_batch["view_indices"].shape[1] // num_video_frames_per_view
        view_indices_B_V_T = rearrange(data_batch["view_indices"], "B (V T) -> B V T", V=n_views)
        latent_view_indices_B_V_T = view_indices_B_V_T[:, :, 0 : self.config.state_t]
        latent_view_indices_B_T = rearrange(latent_view_indices_B_V_T, "B V T -> B (V T)")
        data_batch["latent_view_indices_B_T"] = latent_view_indices_B_T
        return data_batch

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor], with_uncondition: bool = True):
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        self.inplace_compute_text_embeddings_online(
            data_batch,
            use_negative_prompt=with_uncondition,
        )

        data_batch_original = copy.deepcopy(data_batch)

        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        condition_original, uncondition_original = self.conditioner.get_condition_with_negative_prompt(
            data_batch_original
        )
        condition_original = condition_original.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        uncondition_original = uncondition_original.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)

        state_t = int(
            (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor + 1
        )
        if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
            num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
        else:
            num_conditional_frames = None

        condition = condition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        condition_original = condition_original.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        uncondition = uncondition.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        uncondition_original = uncondition_original.set_video_condition(
            state_t=state_t,
            gt_frames=latent_state.to(**self.tensor_kwargs),
            condition_locations=["first_random_n"],
            random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
            num_conditional_frames_per_view=num_conditional_frames,
            view_condition_dropout_max=0,
            conditional_frames_probs=self.config.conditional_frames_probs,
        )

        if with_uncondition:
            return raw_state, latent_state, (condition, uncondition, condition_original, uncondition_original)
        else:
            return raw_state, latent_state, (condition, condition_original)

    def get_x0_fn_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        is_negative_prompt: bool = False,
        conditional_dict: dict = None,
    ):
        assert data_batch is not None or conditional_dict is not None, "data_batch or conditional_dict must be provided"

        if data_batch is not None:
            data_batch = self.get_data_batch_with_latent_view_indices(data_batch)
            if NUM_CONDITIONAL_FRAMES_KEY in data_batch:
                num_conditional_frames = data_batch[NUM_CONDITIONAL_FRAMES_KEY]
            else:
                num_conditional_frames = None

        if conditional_dict is None:
            _, latent_state, _ = self.get_data_and_condition(data_batch, with_uncondition=False)
            is_image_batch = self.is_image_batch(data_batch)

            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)

            condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
            _, condition, _, _ = self.broadcast_split_for_model_parallelsim(None, condition, None, None)

            state_t = int(
                (data_batch["num_video_frames_per_view"].cpu().item() - 1) // self.tokenizer.temporal_compression_factor
                + 1
            )

            condition = condition.set_video_condition(
                state_t=state_t,
                gt_frames=latent_state.to(**self.tensor_kwargs),
                condition_locations=["first_random_n"],
                random_min_num_conditional_frames_per_view=self.config.min_num_conditional_frames,
                random_max_num_conditional_frames_per_view=self.config.max_num_conditional_frames,
                num_conditional_frames_per_view=num_conditional_frames,
                view_condition_dropout_max=0,
                conditional_frames_probs=self.config.conditional_frames_probs,
            )

            extra_cond = self._extract_extra_conditional_kwargs(data_batch)
            conditional_dict = {**condition.to_dict(), **extra_cond}

        if parallel_state.is_initialized():
            pass
        else:
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def x0_fn(
            noise_x: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            i2v_force_add_into_cache: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            assert self.config.model_type == "i2v"

            noise_x = noise_x.permute(0, 2, 1, 3, 4)
            new_condition_dict = copy.deepcopy(conditional_dict)

            if (
                new_condition_dict["gt_frames"] is not None
                and new_condition_dict["gt_frames"].shape[2] != noise_x.shape[2]
            ):
                assert kwargs.get("start_frame_for_rope", None) is not None, "start_frame_for_rope is not provided"
                start_frame = kwargs.get("start_frame_for_rope")
                end_frame = start_frame + noise_x.shape[2]

                new_condition_dict["gt_frames"] = new_condition_dict["gt_frames"][:, :, start_frame:end_frame, :, :]
                if new_condition_dict["condition_video_input_mask_B_C_T_H_W"] is not None:
                    new_condition_dict["condition_video_input_mask_B_C_T_H_W"] = new_condition_dict[
                        "condition_video_input_mask_B_C_T_H_W"
                    ][:, :, start_frame:end_frame, :, :]
                if new_condition_dict["view_indices_B_T"] is not None:
                    new_condition_dict["view_indices_B_T"] = new_condition_dict["view_indices_B_T"][
                        :, start_frame:end_frame
                    ]

            _, denoised_pred = self.generator(
                noisy_image_or_video=noise_x.permute(0, 2, 1, 3, 4),
                conditional_dict=new_condition_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                **kwargs,
            )
            return denoised_pred

        return x0_fn

    def generate_samples_from_batch(
        self,
        data_batch: dict[str, torch.Tensor] | None = None,
        guidance: float = 1.0,
        seed: int = 1,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        start_latents: Optional[torch.Tensor] = None,
        verbose: bool = False,
        conditional_dict: dict | None = None,
        image_or_video_shape: Tuple | None = None,
        noise_B_T_C_H_W: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if data_batch is not None:
            self._normalize_video_databatch_inplace(data_batch)
            self._augment_image_dim_inplace(data_batch)
            is_image_batch = self.is_image_batch(data_batch)
            input_key = self.input_image_key if is_image_batch else self.input_data_key
            if n_sample is None:
                n_sample = data_batch[input_key].shape[0]
            if state_shape is None:
                _T, _H, _W = data_batch[input_key].shape[-3:]
                state_shape = [
                    self.tokenizer.get_latent_num_frames(_T),
                    self.config.state_ch,
                    _H // self.tokenizer.spatial_compression_factor,
                    _W // self.tokenizer.spatial_compression_factor,
                ]
            else:
                state_shape = (state_shape[1], state_shape[0], *state_shape[2:])
        assert state_shape is not None or image_or_video_shape is not None, (
            "data_batch or image_or_video_shape must be provided"
        )
        if noise_B_T_C_H_W is None:
            noise_B_T_C_H_W = misc.arch_invariant_rand(
                (n_sample,) + tuple(state_shape) if image_or_video_shape is None else image_or_video_shape,
                torch.float32,
                self.tensor_kwargs["device"],
                seed,
            )
            misc.set_random_seed(seed=seed, by_rank=False)
        self.frame_seq_length = int(noise_B_T_C_H_W.shape[-1] * noise_B_T_C_H_W.shape[-2] / 4)
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            self.net.enable_context_parallel(cp_group)
            noise_B_T_C_H_W = broadcast(noise_B_T_C_H_W.contiguous(), cp_group)
            if start_latents is not None:
                start_latents = broadcast(start_latents.contiguous(), cp_group)
        else:
            assert not getattr(self.net, "is_context_parallel_enabled", False), (
                "context parallel should be disabled if parallel_state is not initialized"
            )
        flow_pred_fn = self.get_x0_fn_from_batch(
            data_batch=data_batch,
            guidance=guidance,
            is_negative_prompt=is_negative_prompt,
            conditional_dict=conditional_dict,
        )

        def x0_fn(
            noisy_image_or_video: torch.Tensor,
            timestep: torch.Tensor,
            kv_cache: Optional[List[dict]] = None,
            crossattn_cache: Optional[List[dict]] = None,
            current_start: Optional[int] = None,
            current_end: Optional[int] = None,
            start_frame_for_rope: Optional[int] = None,
        ):
            return flow_pred_fn(
                noise_x=noisy_image_or_video,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                current_end=current_end,
                start_frame_for_rope=start_frame_for_rope,
            )

        batch_size, num_frames, num_channels, height, width = noise_B_T_C_H_W.shape
        num_input_frames = 0
        num_output_frames = num_frames + num_input_frames
        output_B_T_C_H_W = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise_B_T_C_H_W.device,
            dtype=noise_B_T_C_H_W.dtype,
        )
        assert num_frames % self.config.num_frame_per_block == 0
        num_blocks = num_frames // self.config.num_frame_per_block
        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=self.tensor_kwargs["dtype"],
            device=self.tensor_kwargs["device"],
            num_cache_frames=num_output_frames,
        )
        self.crossattn_cache = None
        current_start_frame = 0
        all_num_frames = [self.config.num_frame_per_block] * num_blocks
        exit_flags = [len(self.denoising_step_list) - 1] * len(all_num_frames)
        for block_index, current_num_frames in enumerate(all_num_frames):
            current_end_frame = current_start_frame + current_num_frames
            noisy_input = noise_B_T_C_H_W[
                :, current_start_frame - num_input_frames : current_end_frame - num_input_frames
            ]
            denoising_step_list = self.denoising_step_list
            for index, current_timestep in enumerate(denoising_step_list):
                exit_flag = index == exit_flags[0]
                timestep = (
                    torch.ones([batch_size, current_num_frames], device=noise_B_T_C_H_W.device, dtype=torch.int64)
                    * current_timestep
                )
                if not exit_flag:
                    with torch.no_grad():
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length // cp_size,
                            current_end=current_end_frame * self.frame_seq_length // cp_size,
                            start_frame_for_rope=current_start_frame,
                        )
                        next_timestep = denoising_step_list[index + 1]
                        current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                        if cp_size > 1:
                            current_noise = broadcast(current_noise.contiguous(), cp_group)
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            current_noise,
                            next_timestep
                            * torch.ones(
                                [batch_size * current_num_frames], device=noise_B_T_C_H_W.device, dtype=torch.long
                            ),
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    with torch.no_grad():
                        denoised_pred = x0_fn(
                            noisy_image_or_video=noisy_input,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length // cp_size,
                            current_end=current_end_frame * self.frame_seq_length // cp_size,
                            start_frame_for_rope=current_start_frame,
                        )
                    break
            output_B_T_C_H_W[:, current_start_frame:current_end_frame] = denoised_pred
            context_timestep = torch.ones_like(timestep) * self.config.context_noise
            if self.config.context_noise > 0:
                current_noise = torch.randn_like(denoised_pred.flatten(0, 1))
                if cp_size > 1:
                    current_noise = broadcast(current_noise.contiguous(), cp_group)
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    current_noise,
                    context_timestep
                    * torch.ones([batch_size * current_num_frames], device=noise_B_T_C_H_W.device, dtype=torch.long),
                ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                x0_fn(
                    noisy_image_or_video=denoised_pred,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length // cp_size,
                    current_end=current_end_frame * self.frame_seq_length // cp_size,
                    start_frame_for_rope=current_start_frame,
                )
            current_start_frame = current_end_frame
        return output_B_T_C_H_W.permute(0, 2, 1, 3, 4)

    def _initialize_kv_cache(self, batch_size, dtype, device, num_cache_frames=None):

        if num_cache_frames is None:
            num_cache_frames = self.config.num_cache_frames

        local_attn_size = getattr(self.net, "local_attn_size", -1)
        if local_attn_size == -1:
            kv_cache_size = self.frame_seq_length * num_cache_frames
        else:
            if local_attn_size > num_cache_frames:
                raise ValueError(
                    f"local_attn_size {local_attn_size} is larger than num_cache_frames {num_cache_frames}, "
                    f"which is not supported"
                )
            kv_cache_size = self.frame_seq_length * local_attn_size

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            assert kv_cache_size % cp_size == 0, "kv_cache_size must be divisible by cp_size"
            kv_cache_size = kv_cache_size // cp_size

        kv_cache1 = []
        for _ in range(self.net.num_layers):
            kv_cache1.append(
                {
                    "k": torch.zeros(
                        [batch_size, int(kv_cache_size), self.net.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, int(kv_cache_size), self.net.num_heads, 128],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            )

        self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):

        crossattn_cache = []

        for _ in range(self.net.num_layers):
            crossattn_cache.append(
                {
                    "k": torch.zeros([batch_size, 512, self.net.num_heads, 128], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, 512, self.net.num_heads, 128], dtype=dtype, device=device),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device) -> List[int]:
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
            if self.config.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()
