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

from __future__ import annotations

from typing import Dict

import attrs
import torch
from einops import rearrange
from torch import Tensor

from worldfoundry.core.configuration.lazy_config import LazyCall as L
from worldfoundry.core.configuration.lazy_config import LazyDict
from worldfoundry.core.configuration.lazy_config import instantiate as lazy_instantiate
from worldfoundry.core.distributed.context_parallel import broadcast, broadcast_split_tensor
from worldfoundry.core.distributed.fsdp_runtime import hsdp_device_mesh
from worldfoundry.core.distributed.logging import log
from worldfoundry.core.distributed.megatron_compat import parallel_state
from worldfoundry.core.model_loading.inference_model import (
    InferenceModel,
    instantiate_inference_network,
)
from worldfoundry.core.utils import inference_runtime as misc
from worldfoundry.core.utils.torch_utils import count_parameters as count_params
from worldfoundry.runtime.compile_cache import CompilePolicy, compile_module_cached
from worldfoundry.synthesis.visual_generation.gamma_world.text_encoder.encoder import TextEncoder, TextEncoderConfig

IS_PREPROCESSED_KEY = "is_preprocessed"
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


@attrs.define(slots=False)
class DMDInferenceBaseModelConfig:
    net: LazyDict | None = None

    fsdp_shard_size: int = 1
    precision: str = "bfloat16"
    use_torch_compile: bool = False
    input_data_key: str = "video"
    input_image_key: str = "images"

    keep_original_net_dtype: bool = True
    mixed_precision_policy_internal_layers: LazyDict = L(torch.distributed.fsdp.MixedPrecisionPolicy)(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    mixed_precision_policy_root_module: LazyDict = L(torch.distributed.fsdp.MixedPrecisionPolicy)(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        cast_forward_inputs=True,
    )

    tokenizer: LazyDict | None = None
    conditioner: LazyDict | None = None

    text_encoder_class: str = "reason1p1_7B"
    text_encoder_config: TextEncoderConfig = TextEncoderConfig(
        embedding_concat_strategy="full_concat",
        compute_online=True,
        ckpt_path="hf://nvidia/Cosmos-Reason1-7B",
    )
    input_caption_key: str = "ai_caption"
    split_cp_in_model: bool = False


class DMDInferenceBaseModel(InferenceModel):
    def __init__(self, config: DMDInferenceBaseModelConfig):
        super().__init__()
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.device = "cuda"
        self.dtype = self.precision
        self.tensor_kwargs = {"device": self.device, "dtype": self.dtype}

        log.warning(f"DiffusionModel: precision {self.precision}")

        self.setup_data_key()

        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)

        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
            self.mixed_precision_policy_internal_layers = lazy_instantiate(
                config.mixed_precision_policy_internal_layers
            )
            self.mixed_precision_policy_root_module = lazy_instantiate(config.mixed_precision_policy_root_module)
        else:
            self.fsdp_device_mesh = None
            self.mixed_precision_policy_internal_layers = None
            self.mixed_precision_policy_root_module = None

        self.set_up_model()

        self.text_encoder = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.text_encoder = TextEncoder(self.config.text_encoder_config)
        self.input_caption_key = self.config.input_caption_key

        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key
        self.input_image_key = self.config.input_image_key

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent)

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating inference model"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            self.net = instantiate_inference_network(
                config.net,
                device_mesh=self.fsdp_device_mesh,
                root_mixed_precision_policy=self.mixed_precision_policy_root_module,
                internal_mixed_precision_policy=self.mixed_precision_policy_internal_layers,
                shard_before_materialize=True,
            )
            self._param_count = count_params(self.net, verbose=False)
        torch.cuda.empty_cache()

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:

        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_data_key should be present in the data_batch."
        )
        return is_image

    def prepare_inference(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        kwargs = {"device": self.device} if self.config.keep_original_net_dtype else self.tensor_kwargs
        self.net = self.net.to(memory_format=memory_format, **kwargs)

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:
            if torch.__version__ < "2.3":
                log.warning("torch.compile performance may be limited on PyTorch older than 2.3")

            torch._dynamo.config.accumulated_cache_size_limit = 256

            self.net = compile_module_cached(
                self.net,
                policy=CompilePolicy(dynamic=False),
                namespace="gamma-world-distilled-dit",
            )

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def inplace_compute_text_embeddings_online(
        self,
        data_batch: dict[str, torch.Tensor],
        use_negative_prompt: bool = True,
        negative_prompt: str = _DEFAULT_NEGATIVE_PROMPT,
    ) -> None:

        if (
            self.config.text_encoder_config is not None
            and self.config.text_encoder_config.compute_online
            and self.text_encoder is not None
        ):
            text_embeddings = self.text_encoder.compute_text_embeddings_online(data_batch, self.input_caption_key)
            data_batch["t5_text_embeddings"] = text_embeddings
            data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device="cuda")

            if use_negative_prompt:
                batch_size = text_embeddings.shape[0]
                neg_data_batch = {self.input_caption_key: [negative_prompt] * batch_size, "images": None}
                neg_text_embeddings = self.text_encoder.compute_text_embeddings_online(
                    neg_data_batch, self.input_caption_key
                )
                data_batch["neg_t5_text_embeddings"] = neg_text_embeddings

    def _convert_flow_pred_to_x0(
        self, scheduler, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:

        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    def _convert_x0_to_flow_pred(
        self, scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:

        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]):
        raise NotImplementedError("DMDModel: get_data_and_condition is not implemented")

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:

        input_key = self.input_data_key if input_key is None else input_key

        if input_key in data_batch:
            _flag = data_batch.get(IS_PREPROCESSED_KEY, False)
            if isinstance(_flag, torch.Tensor):
                try:
                    _flag = bool(_flag.bool().all().item())
                except Exception:
                    _flag = False
            else:
                _flag = bool(_flag)

            if _flag:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

    def _augment_image_dim_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        input_key = self.input_image_key if input_key is None else input_key
        if input_key in data_batch:
            _flag = data_batch.get(IS_PREPROCESSED_KEY, False)
            if isinstance(_flag, torch.Tensor):
                try:
                    _flag = bool(_flag.bool().all().item())
                except Exception:
                    _flag = False
            else:
                _flag = bool(_flag)

            if _flag:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T, split=False):

        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group, split=split)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
