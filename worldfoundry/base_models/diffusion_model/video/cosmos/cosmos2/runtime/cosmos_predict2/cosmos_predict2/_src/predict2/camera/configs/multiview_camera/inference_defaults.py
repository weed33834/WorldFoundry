# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight camera-conditioned inference configuration."""

from cosmos_predict2._src.predict2.configs.video2world.inference_defaults import register_inference_defaults
from cosmos_predict2._src.predict2.configs.video2world.released import released_config
from hydra.core.config_store import ConfigStore

from worldfoundry.core.configuration.lazy_config import LazyCall as L

CAMERA_EXPERIMENT = "multicamera_video2video_rectified_flow_2b_res_720_fps16_s3_agibot_frameinit"


def create_camera_model(config: dict, variant: str = "frameinit"):
    if variant == "frameinit":
        from cosmos_predict2._src.predict2.camera.models.multiview_camera_frameinit_video2world_model import (
            CameraConditionedFrameinitVideo2WorldModelRectifiedFlow as Model,
        )
        from cosmos_predict2._src.predict2.camera.models.multiview_camera_frameinit_video2world_model import (
            CameraConditionedFrameinitVideo2WorldRectifiedFlowConfig as ModelConfig,
        )
    elif variant == "autoregressive":
        from cosmos_predict2._src.predict2.camera.models.multiview_camera_ar_video2world_model import (
            CameraConditionedARVideo2WorldModelRectifiedFlow as Model,
        )
        from cosmos_predict2._src.predict2.camera.models.multiview_camera_ar_video2world_model import (
            CameraConditionedARVideo2WorldRectifiedFlowConfig as ModelConfig,
        )
    else:
        from cosmos_predict2._src.predict2.camera.models.multiview_camera_video2world_model import (
            CameraConditionedVideo2WorldModelRectifiedFlow as Model,
        )
        from cosmos_predict2._src.predict2.camera.models.multiview_camera_video2world_model import (
            CameraConditionedVideo2WorldRectifiedFlowConfig as ModelConfig,
        )
    return Model(ModelConfig(**config))


def _camera_net(*, autoregressive: bool = False) -> dict:
    module = "dit_multiview_camera_ar" if autoregressive else "dit_multiview_camera"
    cls = "CameraARInferenceDiTWithConditionalMask" if autoregressive else "CameraInferenceDiTWithConditionalMask"
    return L(f"cosmos_predict2._src.predict2.camera.networks.{module}.{cls}")(
        max_img_h=240,
        max_img_w=240,
        max_frames=128,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        model_channels=2048,
        num_blocks=28,
        num_heads=16,
        concat_padding_mask=True,
        pos_emb_cls="rope3d",
        pos_emb_learnable=True,
        pos_emb_interpolation="crop",
        use_adaln_lora=True,
        adaln_lora_dim=256,
        atten_backend="minimal_a2a",
        extra_per_block_abs_pos_emb=False,
        rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
        rope_t_extrapolation_ratio=1.0,
        sac_config=L("cosmos_predict2._src.predict2.camera.networks.dit_multiview_camera.SACConfig")(),
    )


def _camera_conditioner(target: str) -> dict:
    remap = "cosmos_predict2._src.predict2.conditioner.ReMapkey"
    common = {
        "fps": L(remap)(input_key="fps", output_key="fps", dropout_rate=0.0, dtype=None),
        "padding_mask": L(remap)(input_key="padding_mask", output_key="padding_mask", dropout_rate=0.0, dtype=None),
        "text": L("cosmos_predict2._src.predict2.conditioner.TextAttr")(
            input_key=["t5_text_embeddings"], dropout_rate=0.2, use_empty_string=False
        ),
        "use_video_condition": L("cosmos_predict2._src.predict2.conditioner.BooleanFlag")(
            input_key="fps", output_key="use_video_condition", dropout_rate=0.2
        ),
        "camera": L("cosmos_predict2._src.predict2.camera.configs.multiview_camera.conditioner.CameraToPluckerRays")(
            extrinsics_key="extrinsics",
            intrinsics_key="intrinsics",
            image_size_key="image_size",
            output_key="camera",
            patch_spatial=16,
            camera_patch_average=False,
            out_dtype="bfloat16",
            dropout_rate=0.0,
        ),
    }
    return L(target)(**common)


def register_camera_inference() -> None:
    register_inference_defaults()
    store = ConfigStore.instance()
    model_module = (
        "cosmos_predict2._src.predict2.camera.configs.multiview_camera.inference_defaults.create_camera_model"
    )
    for name, variant in (
        ("camera_model", "standard"),
        ("camera_frameinit_model", "frameinit"),
        ("camera_ar_model", "autoregressive"),
    ):
        store.store(
            group="model",
            package="_global_",
            name=name,
            node={"model": L(model_module)(config={"fsdp_shard_size": 8}, variant=variant, _recursive_=False)},
        )
    store.store(group="net", package="model.config.net", name="camera_net", node=_camera_net())
    store.store(group="net", package="model.config.net", name="camera_ar_net", node=_camera_net(autoregressive=True))
    conditioner_module = "cosmos_predict2._src.predict2.camera.configs.multiview_camera.conditioner"
    for name, cls in (
        ("camera_conditioner", "CameraConditionedConditioner"),
        ("camera_frameinit_conditioner", "CameraConditionedFrameinitConditioner"),
        ("camera_ar_conditioner", "CameraConditionedARConditioner"),
    ):
        store.store(
            group="conditioner",
            package="model.config.conditioner",
            name=name,
            node=_camera_conditioner(f"{conditioner_module}.{cls}"),
        )

    config = released_config(size="2B", name=CAMERA_EXPERIMENT)
    config["defaults"] = [
        {"override /model": "camera_frameinit_model"},
        {"override /net": "camera_net"},
        {"override /conditioner": "camera_frameinit_conditioner"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /checkpoint": "s3"},
        "_self_",
    ]
    config["model_parallel"] = {"context_parallel_size": 8}
    store.store(group="experiment", package="_global_", name=CAMERA_EXPERIMENT, node=config)


__all__ = ["CAMERA_EXPERIMENT", "register_camera_inference"]
