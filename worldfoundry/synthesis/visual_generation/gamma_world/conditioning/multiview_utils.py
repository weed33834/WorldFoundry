# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-view prompt embedding helpers used by Gamma inference."""

from typing import Tuple

import torch
from einops import rearrange

IS_PREPROCESSED_KEY = "is_preprocessed"
_DEFAULT_NEGATIVE_PROMPT = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."


def compute_text_embeddings_online_multiview_single_caption(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    B, _, _, _, _ = data_batch[model.input_data_key].shape

    # compute prompt embeddings
    if len(data_batch["ai_caption"]) != 1:
        raise NotImplementedError(f"Expected batch size of 1, got {len(data_batch['ai_caption'])}")

    if len(data_batch["ai_caption"][0]) != 1:
        raise ValueError(f"Expected a single caption, got {len(data_batch['ai_caption'][0])}")

    caption = data_batch["ai_caption"][0][0]
    assert isinstance(caption, str)
    view0_text_embeddings_B_L_D = model.text_encoder.compute_text_embeddings_online(
        data_batch={model.input_caption_key: [caption]},
        input_caption_key=model.input_caption_key,
    )
    assert view0_text_embeddings_B_L_D.shape[0] == 1
    assert view0_text_embeddings_B_L_D.shape[1] == 512, (
        f"view0_text_embeddings should be of shape (B, 512, D), got {view0_text_embeddings_B_L_D.shape}"
    )
    output_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    output_neg_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    output_text_embeddings = rearrange(output_text_embeddings, "B (V L) D -> V B L D", V=n_views)
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings, "B (V L) D -> V B L D", V=n_views)
    # Assign prompt embeddings to the front camera view
    for i_b in range(B):
        front_cam_view_idx_sample_position = data_batch["front_cam_view_idx_sample_position"][i_b]
        output_text_embeddings[front_cam_view_idx_sample_position, i_b] = view0_text_embeddings_B_L_D[i_b]
        output_neg_text_embeddings[front_cam_view_idx_sample_position, i_b] = model.neg_text_embeddings[0]
    output_text_embeddings = rearrange(output_text_embeddings, "V B L D -> B (V L) D")
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings, "V B L D -> B (V L) D")

    dropout_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)

    if not model.config.conditioner.text.use_empty_string:
        dropout_text_embeddings *= 0.0

    return output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings


def compute_text_embeddings_online_multiview_multiple_captions(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    B, _, _, _, _ = data_batch[model.input_data_key].shape

    # compute each view's caption separately
    if not len(data_batch["ai_caption"]) == 1:
        raise NotImplementedError(f"Expected batch size of 1, got {len(data_batch['ai_caption'])}")

    captions = data_batch["ai_caption"][0]
    if len(captions) != n_views:
        raise ValueError(f"Expected {n_views} captions, got {len(captions)}: {captions}")
    view_text_embeddings = []
    for caption in captions:
        data_batch_per_view = {model.input_caption_key: [caption]}
        view_text_embedding = model.text_encoder.compute_text_embeddings_online(
            data_batch_per_view, model.input_caption_key
        )
        view_text_embeddings.append(view_text_embedding)

    view_text_embeddings_B_V_L_D = torch.stack(view_text_embeddings, dim=1)
    assert view_text_embeddings_B_V_L_D.shape[:3] == (
        B,
        n_views,
        512,
    ), f"view_text_embeddings_B_V_L_D should be of shape (B, n_views, 512, D), got {view_text_embeddings_B_V_L_D.shape}"
    output_text_embeddings = rearrange(view_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")

    # repeat negative embedding for each per view
    output_neg_text_embeddings_L_D = model.neg_text_embeddings[0].clone()
    output_neg_text_embeddings_B_V_L_D = rearrange(output_neg_text_embeddings_L_D, "(B V L) D -> B V L D", B=1, V=1)
    output_neg_text_embeddings_B_V_L_D = output_neg_text_embeddings_B_V_L_D.repeat(B, n_views, 1, 1)
    assert output_neg_text_embeddings_B_V_L_D.shape[:3] == (
        B,
        n_views,
        512,
    ), (
        f"output_neg_text_embeddings_B_V_L_D should be of shape (B, n_views, 512, D), got {output_neg_text_embeddings_B_V_L_D.shape}"
    )
    output_neg_text_embeddings = rearrange(output_neg_text_embeddings_B_V_L_D, "B V L D -> B (V L) D")

    dropout_text_embeddings = model.empty_string_text_embeddings.clone().repeat(B, n_views, 1)
    if not model.config.conditioner.text.use_empty_string:
        dropout_text_embeddings *= 0.0

    return output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings


def compute_text_embeddings_online_multiview(
    model, data_batch: dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    captions = data_batch["ai_caption"][0]
    is_preprocessed = IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True
    num_video_frames_per_view = (
        model.tokenizer.get_pixel_num_frames(model.state_t)
        if is_preprocessed
        else data_batch["num_video_frames_per_view"]
    )
    if isinstance(num_video_frames_per_view, torch.Tensor):
        num_video_frames_per_view = int(num_video_frames_per_view.cpu().item())
    n_views = data_batch[model.input_data_key].shape[2] // num_video_frames_per_view
    assert len(captions) == 1 or len(captions) == n_views, f"Expected 1 or {n_views} captions, got {len(captions)}"
    if len(captions) == 1:
        return compute_text_embeddings_online_multiview_single_caption(model, data_batch)
    else:
        return compute_text_embeddings_online_multiview_multiple_captions(model, data_batch)


def inplace_compute_text_embeddings_online_multiview(model, data_batch: dict[str, torch.Tensor]) -> None:
    output_text_embeddings, output_neg_text_embeddings, dropout_text_embeddings = (
        compute_text_embeddings_online_multiview(model, data_batch)
    )
    t5_text_embeddings = {
        "text_embeddings": output_text_embeddings,
        "dropout_text_embeddings": dropout_text_embeddings,
    }
    neg_t5_text_embeddings = {
        "text_embeddings": output_neg_text_embeddings,
        "dropout_text_embeddings": dropout_text_embeddings,
    }
    data_batch["t5_text_embeddings"] = (
        t5_text_embeddings if model.config.online_text_embeddings_as_dict else t5_text_embeddings["text_embeddings"]
    )
    data_batch["neg_t5_text_embeddings"] = (
        neg_t5_text_embeddings
        if model.config.online_text_embeddings_as_dict
        else neg_t5_text_embeddings["text_embeddings"]
    )
    data_batch["t5_text_mask"] = torch.ones(
        output_text_embeddings.shape[0], output_text_embeddings.shape[1], device="cuda"
    )


def compute_empty_and_negative_text_embeddings(model):
    # Compute empty string embeddings for text embedding dropout
    if model.empty_string_text_embeddings is None:
        empty_string_data_batch = {
            model.input_caption_key: [" "],
        }
        model.empty_string_text_embeddings = model.text_encoder.compute_text_embeddings_online(
            empty_string_data_batch, model.input_caption_key
        )

    # compute negative prompt embeddings for sampling
    if model.neg_text_embeddings is None:
        neg_promt_data_batch = {
            model.input_caption_key: [_DEFAULT_NEGATIVE_PROMPT],
        }
        model.neg_text_embeddings = model.text_encoder.compute_text_embeddings_online(
            neg_promt_data_batch, model.input_caption_key
        )
