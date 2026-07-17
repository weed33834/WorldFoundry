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

import torch
from transformers import AutoConfig, Qwen2Model

from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.base import VLMBaseModel, init_mesh
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.config import FSDP2ModelConfig
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.parallel import parallelize_qwen
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.qwen2 import (
    Qwen2VisionTransformerPretrainedModel,
    Qwen2VLModel,
)
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.qwen2_5 import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from worldfoundry.base_models.llm_mllm_core.mllm.qwen.cosmos_reason1.inference.tokenizer import Processor


class QwenModel(VLMBaseModel):
    """Qwen backbone used only to extract multimodal hidden states."""

    def __init__(
        self,
        model_config: FSDP2ModelConfig,
        tokenizer: Processor,
    ) -> "QwenModel":
        super().__init__(model_config, tokenizer)
        self.forward_time = []

    def build_model(self, model_config):
        if model_config.model_type == "qwen2_5_vl":
            self.visual = Qwen2_5_VisionTransformerPretrainedModel(model_config.vision_config)
            self.model = Qwen2_5_VLModel(model_config)
        elif model_config.model_type == "qwen2_vl":
            self.visual = Qwen2VisionTransformerPretrainedModel(model_config.vision_config)
            self.model = Qwen2VLModel(model_config)
        elif model_config.model_type == "qwen2_5":
            self.visual = None
            config = AutoConfig.from_pretrained(
                model_config.name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
            )
            self.model = Qwen2Model(config)
            model_config.hidden_size = config.hidden_size
            model_config.vocab_size = config.vocab_size
            self.model.set_cp_mesh = lambda x: None
            self.model.cp_mesh = None
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        self.rope_deltas = None  # cache rope_deltas here]

        if torch.distributed.is_initialized():
            self.world_mesh, self.parallel_dims = init_mesh(model_config)
            parallelize_qwen(self, self.world_mesh, self.parallel_dims, model_config)
            self.model.set_cp_mesh(self.cp_mesh)

    @property
    def vision_encoder(self):
        # This is to be compatible with VLMBaseModel
        return self.visual

    @property
    def mm_projector(self):
        # This is to be compatible with VLMBaseModel
        if self.vision_encoder is not None:
            return self.visual.merger
        else:
            return None

    @property
    def cp_mesh(self):
        if not torch.distributed.is_initialized():
            return None
        # when none of the parallelisms are enabled, the world_mesh.mesh_dim_names is None
        if self.world_mesh.mesh_dim_names is not None and "cp" in self.world_mesh.mesh_dim_names:
            return self.world_mesh["cp"]
        else:
            return None

    @property
    def tp_mesh(self):
        if not torch.distributed.is_initialized():
            return None
        # when none of the parallelisms are enabled, the world_mesh.mesh_dim_names is None
        if self.world_mesh.mesh_dim_names is not None and "tp" in self.world_mesh.mesh_dim_names:
            return self.world_mesh["tp"]
        else:
            return None
