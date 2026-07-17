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

import os
from typing import Optional, Tuple

import attrs
import torch
from einops import rearrange
from torch import Tensor

from worldfoundry.core.configuration.lazy_config import LazyDict
from worldfoundry.core.configuration.lazy_config import instantiate as lazy_instantiate
from worldfoundry.core.distributed.context_parallel import broadcast, broadcast_split_tensor, find_split
from worldfoundry.core.distributed.fsdp_runtime import hsdp_device_mesh
from worldfoundry.core.distributed.logging import log
from worldfoundry.core.distributed.megatron_compat import parallel_state
from worldfoundry.core.io.resolutions import VIDEO_RES_SIZE_INFO
from worldfoundry.core.model_loading.inference_model import (
    InferenceModel,
    instantiate_inference_network,
)
from worldfoundry.core.utils import inference_runtime as misc
from worldfoundry.core.utils.torch_utils import count_parameters as count_params
from worldfoundry.runtime.compile_cache import CompilePolicy, compile_module_cached
from worldfoundry.synthesis.visual_generation.gamma_world.conditioning.base import DataType, Text2WorldCondition
from worldfoundry.synthesis.visual_generation.gamma_world.schedulers.unipc import FlowUniPCMultistepScheduler
from worldfoundry.synthesis.visual_generation.gamma_world.text_encoder.encoder import TextEncoder, TextEncoderConfig

IS_PREPROCESSED_KEY = "is_preprocessed"


@attrs.define(slots=False)
class Text2WorldModelRectifiedFlowConfig:
    """
    Config for [DiffusionModel][projects.cosmos.diffusion.v2.models.text2world_model.DiffusionModel].
    """

    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    fsdp_shard_size: int = 1
    precision: str = "bfloat16"
    input_data_key: str = "video"  # key to fetch input data from data_batch
    input_image_key: str = "images"  # key to fetch input image from data_batch
    input_caption_key: str = "ai_caption"  # Key used to fetch input captions
    use_torch_compile: bool = False

    state_ch: int = 16  # for latent model, ref to the latent channel number
    state_t: int = 8  # for latent model, ref to the latent number of frames
    resolution: str = "512"

    text_encoder_class: str = "T5"
    text_encoder_config: Optional[TextEncoderConfig] = None
    shift: int = 5
    use_kerras_sigma_at_inference: bool = False  # if True, override unipc's timestep schedule with kerras schedule

    def __attrs_post_init__(self):
        assert self.text_encoder_class in ["T5", "umT5", "reason1_2B", "reason1_7B", "reason1p1_7B"]


class Text2WorldModelRectifiedFlow(InferenceModel):
    """
    Diffusion model.
    """

    def __init__(self, config: Text2WorldModelRectifiedFlowConfig):
        super().__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.tensor_kwargs_fp32 = {"device": "cuda", "dtype": torch.float32}
        log.warning(f"DiffusionModel: precision {self.precision}")

        # 1. set data keys and data information
        self.sigma_data = 1.0
        self.sigma_conditional = 0.0001
        self.change_time_embed = False
        self.setup_data_key()

        # 2. setup up rectified_flow and sampler
        self.sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )

        # 3. tokenizer
        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == self.config.state_ch, (
                f"latent_ch {self.tokenizer.latent_ch} != state_shape {self.config.state_ch}"
            )

        # 4. create fsdp mesh if needed
        if config.fsdp_shard_size > 1:
            self.fsdp_device_mesh = hsdp_device_mesh(
                sharding_group_size=config.fsdp_shard_size,
            )
        else:
            self.fsdp_device_mesh = None

        # 5. diffusion neural networks part
        self.set_up_model()

        # 6. text encoder
        self.text_encoder = None
        if self.config.text_encoder_config is not None and self.config.text_encoder_config.compute_online:
            self.text_encoder = TextEncoder(self.config.text_encoder_config)

        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    def setup_data_key(self) -> None:
        self.input_data_key = self.config.input_data_key  # by default it is video key for Video diffusion model
        self.input_image_key = self.config.input_image_key
        self.input_caption_key = self.config.input_caption_key

    def build_net(self, keep_on_cpu: bool = False):
        with misc.timer("Creating PyTorch model"):
            net = instantiate_inference_network(
                self.config.net,
                device="cpu" if keep_on_cpu else "cuda",
                device_mesh=None if keep_on_cpu else self.fsdp_device_mesh,
            )
            self._param_count = count_params(net, verbose=False)
        if int(os.environ.get("COSMOS_PREDICT2_OFFLOAD_DIT", "0")) > 0:
            net.cpu()
        return net

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch inference model"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )
            self.net = self.build_net()
            self._param_count = count_params(self.net, verbose=False)

        torch.cuda.empty_cache()

    def prepare_inference(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)

        if hasattr(self.config, "use_torch_compile") and self.config.use_torch_compile:  # compatible with old config
            if torch.__version__ < "2.3":
                log.warning("torch.compile performance may be limited on PyTorch older than 2.3")
            # Increasing cache size. It's required because of the model size and dynamic input shapes resulting in
            # multiple different triton kernels. For 28 TransformerBlocks, the cache limit of 256 should be enough for
            # up to 9 different input shapes, as 28*9 < 256. If you have more Blocks or input shapes, and you observe
            # graph breaks at each Block (detectable with torch._dynamo.explain) or warnings about
            # exceeding cache limit, you may want to increase this size.
            # Starting with 24.05 Pytorch container, the default value is 256 anyway.
            # You can read more about it in the comments in Pytorch source code under path torch/_dynamo/cache_size.py.
            torch._dynamo.config.accumulated_cache_size_limit = 256
            # dynamic=False means that a separate kernel is created for each shape. It incurs higher compilation costs
            # at initial iterations, but can result in more specialized and efficient kernels.
            # dynamic=True currently throws errors in pytorch 2.3.
            self.net = compile_module_cached(
                self.net,
                policy=CompilePolicy(dynamic=False),
                namespace="gamma-world-dit",
            )

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def broadcast_split_for_model_parallelsim(self, x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T):
        """
        Broadcast and split the input data and condition for model parallelism.
        Currently, we only support context parallelism.
        """
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if condition.is_video and cp_size > 1:
            # Perform spatial split only when it's required, i.e. temporal split is not enough.
            # Refer to "find_split" definition for more details.
            use_spatial_split = cp_size > x0_B_C_T_H_W.shape[2] or x0_B_C_T_H_W.shape[2] % cp_size != 0
            after_split_shape = find_split(x0_B_C_T_H_W.shape, cp_size) if use_spatial_split else None
            if use_spatial_split:
                x0_B_C_T_H_W = rearrange(x0_B_C_T_H_W, "B C T H W -> B C (T H W)")
                if epsilon_B_C_T_H_W is not None:
                    epsilon_B_C_T_H_W = rearrange(epsilon_B_C_T_H_W, "B C T H W -> B C (T H W)")
            x0_B_C_T_H_W = broadcast_split_tensor(x0_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            epsilon_B_C_T_H_W = broadcast_split_tensor(epsilon_B_C_T_H_W, seq_dim=2, process_group=cp_group)
            if use_spatial_split:
                x0_B_C_T_H_W = rearrange(
                    x0_B_C_T_H_W, "B C (T H W) -> B C T H W", T=after_split_shape[0], H=after_split_shape[1]
                )
                if epsilon_B_C_T_H_W is not None:
                    epsilon_B_C_T_H_W = rearrange(
                        epsilon_B_C_T_H_W, "B C (T H W) -> B C T H W", T=after_split_shape[0], H=after_split_shape[1]
                    )
            if sigma_B_T is not None:
                assert sigma_B_T.ndim == 2, "sigma_B_T should be 2D tensor"
                if sigma_B_T.shape[-1] == 1:  # single sigma is shared across all frames
                    sigma_B_T = broadcast(sigma_B_T, cp_group)
                else:  # different sigma for each frame
                    sigma_B_T = broadcast_split_tensor(sigma_B_T, seq_dim=1, process_group=cp_group)
            if condition is not None:
                condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        return x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T

    # ------------------------ Sampling ------------------------

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, Text2WorldCondition]:
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)

        # Latent state
        raw_state = data_batch[self.input_image_key if is_image_batch else self.input_data_key]
        latent_state = self.encode(raw_state).contiguous().float()

        # Condition
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE if is_image_batch else DataType.VIDEO)
        return raw_state, latent_state, condition

    def _normalize_video_databatch_inplace(self, data_batch: dict[str, Tensor], input_key: str = None) -> None:
        """
        Normalizes video data in-place on a CUDA device to reduce data loading overhead.

        This function modifies the video data tensor within the provided data_batch dictionary
        in-place, scaling the uint8 data from the range [0, 255] to the normalized range [-1, 1].

        Warning:
            A warning is issued if the data has not been previously normalized.

        Args:
            data_batch (dict[str, Tensor]): A dictionary containing the video data under a specific key.
                This tensor is expected to be on a CUDA device and have dtype of torch.uint8.

        Side Effects:
            Modifies the 'input_data_key' tensor within the 'data_batch' dictionary in-place.

        Note:
            This operation is performed directly on the CUDA device to avoid the overhead associated
            with moving data to/from the GPU. Ensure that the tensor is already on the appropriate device
            and has the correct dtype (torch.uint8) to avoid unexpected behaviors.
        """
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
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
            # Check if the data has already been augmented and avoid re-augmenting
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert data_batch[input_key].shape[2] == 1, (
                    f"Image data is claimed be augmented while its shape is {data_batch[input_key].shape}"
                )
                return
            else:
                data_batch[input_key] = rearrange(data_batch[input_key], "b c h w -> b c 1 h w").contiguous()
                data_batch[IS_PREPROCESSED_KEY] = True

    # ------------------ Checkpointing ------------------

    # ------------------ public methods ------------------

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        """Return whether the inference batch contains an image or a video."""
        is_image = self.input_image_key in data_batch
        is_video = self.input_data_key in data_batch
        assert is_image != is_video, (
            "Only one of the input_image_key or input_data_key should be present in the data_batch."
        )
        return is_image

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent)

    def get_video_height_width(self) -> Tuple[int, int]:
        return VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]

    def get_video_latent_height_width(self) -> Tuple[int, int]:
        height, width = VIDEO_RES_SIZE_INFO[self.config.resolution]["9,16"]
        return height // self.tokenizer.spatial_compression_factor, width // self.tokenizer.spatial_compression_factor

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class
