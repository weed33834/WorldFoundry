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

"""
Action-conditioned video generation inference.
"""

import os
from pathlib import Path

import mediapy
import numpy as np
import torch
import torchvision
from cosmos_predict2._src.predict2.action.action_utils import get_action_sequence_from_states
from cosmos_predict2._src.predict2.inference.video2world import (
    Video2WorldInference,
)
from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, VIDEO_EXTENSIONS
from groot_dreams.eval_inputsloader import MultiVideoActionDataset
from loguru import logger

from worldfoundry.core.distributed import torch_process_group as distributed


def load_default_action_fn():
    """
    Default action loading function that processes robot states into actions.
    """

    def load_fn(
        json_data: dict,
        video_path: str,
        args: ActionConditionedInferenceArguments,
    ) -> dict:
        """
        Load action data from JSON and prepare it for inference.

        Args:
            json_data: JSON data containing robot states
            video_path: Path to the video file
            args: Inference arguments

        Returns:
            Dictionary containing actions, video data, and metadata
        """
        # Get action sequence from states
        actions = get_action_sequence_from_states(
            json_data,
            fps_downsample_ratio=args.fps_downsample_ratio,
            state_key=args.state_key,
            gripper_scale=args.gripper_scale,
            gripper_key=args.gripper_key,
            action_scale=args.action_scaler,
            use_quat=args.use_quat,
        )

        # Load video
        video_array = mediapy.read_video(video_path)
        img_array = video_array[args.start_frame_idx]

        # Resize if specified
        if args.resolution != "none":
            try:
                h, w = map(int, args.resolution.split(","))
                img_array = mediapy.resize_image(img_array, (h, w))
            except Exception as e:
                logger.warning(f"Failed to resize image to {args.resolution}: {e}")

        return {
            "actions": actions,
            "initial_frame": img_array,
            "video_array": video_array,
            "video_path": video_path,
        }

    return load_fn


def load_callable(name: str):
    """Load a callable function from a module path string."""
    from importlib import import_module

    idx = name.rfind(".")
    assert idx > 0, "expected <module_name>.<identifier>"
    module_name = name[0:idx]
    fn_name = name[idx + 1 :]

    module = import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


def get_video_id(img_path: str):
    """Extract video ID from image path by removing directory and extension."""
    return img_path.split("/")[-1].split(".")[0]


def inference(
    setup_args: ActionConditionedSetupArguments,
    inference_args,
    checkpoint_path,
):
    """Run action-conditioned video generation inference using resolved setup and per-run arguments."""
    torch.enable_grad(False)  # Disable gradient calculations for inference

    # Validate num_latent_conditional_frames at the very beginning
    if inference_args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            f"num_latent_conditional_frames must be 0, 1 or 2, but got {inference_args.num_latent_conditional_frames}"
        )

    # Determine supported extensions based on num_latent_conditional_frames
    if inference_args.num_latent_conditional_frames > 1:
        # Check if input folder contains any videos
        has_videos = False
        for file_name in os.listdir(inference_args.input_root):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in VIDEO_EXTENSIONS:
                has_videos = True
                break

        if not has_videos:
            raise ValueError(
                f"num_latent_conditional_frames={inference_args.num_latent_conditional_frames} > 1 requires video inputs, "
                f"but no videos found in {inference_args.input_root}. Found extensions: "
                f"{set(os.path.splitext(f)[1].lower() for f in os.listdir(inference_args.input_root) if os.path.splitext(f)[1])}"
            )

        logger.info(f"Using video-only mode with {inference_args.num_latent_conditional_frames} conditional frames")
    elif inference_args.num_latent_conditional_frames == 1:
        logger.info(f"Using image+video mode with {inference_args.num_latent_conditional_frames} conditional frame")

    # Get checkpoint and experiment from setup args
    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    # pyrefly: ignore  # missing-attribute
    # checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

    # Ensure experiment is not None
    if experiment is None:
        raise ValueError("Experiment name must be provided either in setup args or checkpoint metadata")

    # Initialize the inference handler with context parallel support
    video2world_cli = Video2WorldInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        # pyrefly: ignore  # bad-argument-type
        context_parallel_size=setup_args.context_parallel_size,
        config_file=setup_args.config_file,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    # Ensure save directory exists
    inference_args.save_root = Path(setup_args.save_dir) / checkpoint_path.parent.name
    inference_args.save_root.mkdir(parents=True, exist_ok=True)

    num_frames = setup_args.num_frames
    num_samples = setup_args.num_samples
    dataset = MultiVideoActionDataset(
        num_frames=num_frames,
        dataset_path=setup_args.dataset_path,
        data_split=setup_args.data_split,
        single_base_index=setup_args.single_base_index,
        restrict_len=num_samples,
        deterministic_uniform_sampling=setup_args.deterministic_uniform_sampling,
    )
    # eval_indices = list(range(0, len(dataset), max(len(dataset) // num_samples, 1)))[:num_samples]
    eval_indices = [
        idx for idx in range(len(dataset)) if not (inference_args.save_root / f"{idx:04d}_psnr.json").exists()
    ]

    # Process each file in the input directory
    for idx in eval_indices:
        data = dataset[idx]
        img_array = data["video"].transpose(0, 1)[:1]
        actions = data["action"][: num_frames - 1].numpy()
        lam_video = data["lam_video"]

        if inference_args.zero_actions:
            actions = np.zeros_like(actions)

        chunk_video = []
        save_name = f"{idx:04d}"

        chunk_video_name = str(inference_args.save_root / f"{save_name}_pred.mp4")
        logger.info(f"Saving video to {chunk_video_name}")
        if os.path.exists(chunk_video_name):
            logger.info(f"Video already exists: {chunk_video_name}")
            continue

        first_round = True
        for i in range(inference_args.start_frame_idx, len(actions), inference_args.chunk_size):
            # Handle incomplete chunks
            actions_chunk = actions[i : i + inference_args.chunk_size]
            assert actions_chunk.shape[0] == inference_args.chunk_size
            # if actions_chunk.shape[0] != inference_args.chunk_size:
            #     pad_len = inference_args.chunk_size - actions_chunk.shape[0]
            #     if pad_len > 0:
            #         action_shape = list(actions.shape[1:])
            #         pad_shape = [pad_len] + action_shape
            #         pad_actions = np.zeros(pad_shape, dtype=actions.dtype)
            #         actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

            current_lam_video = lam_video[i * 2 : (i + inference_args.chunk_size) * 2]

            # Convert img_array to tensor and prepare video input
            # pyrefly: ignore  # implicit-import
            if not first_round:
                img_tensor = torchvision.transforms.functional.to_tensor(img_array).unsqueeze(0) * 255.0
            else:
                img_tensor = img_array
            first_round = False
            num_video_frames = actions_chunk.shape[0] + 1
            vid_input = torch.cat(
                [img_tensor, torch.zeros_like(img_tensor).repeat(num_video_frames - 1, 1, 1, 1)], dim=0
            )
            vid_input = vid_input.to(torch.uint8)
            vid_input = vid_input.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)

            # Call generate_vid2world
            video = video2world_cli.generate_vid2world(
                prompt="",
                input_path=vid_input,
                action=torch.from_numpy(actions_chunk).float()
                if isinstance(actions_chunk, np.ndarray)
                else actions_chunk,
                guidance=inference_args.guidance,
                num_video_frames=num_video_frames,
                num_latent_conditional_frames=inference_args.num_latent_conditional_frames,
                resolution="480,640",
                seed=i,
                negative_prompt=inference_args.negative_prompt,
                lam_video=current_lam_video,
            )
            # Extract next frame and video from result
            video_normalized = (video - (-1)) / (1 - (-1))
            video_clamped = (
                (torch.clamp(video_normalized[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            )
            next_img_array = video_clamped[-1]  # Last frame is the next frame
            img_array = next_img_array
            chunk_video.append(video_clamped)

            if inference_args.single_chunk:
                break

        chunk_list = [chunk_video[0]] + [
            chunk_video[i][: inference_args.chunk_size] for i in range(1, len(chunk_video))
        ]
        chunk_video = np.concatenate(chunk_list, axis=0)
        chunk_video_name = str(inference_args.save_root / f"{save_name}_pred.mp4")

        if rank0:
            mediapy.write_video(chunk_video_name, chunk_video, fps=inference_args.save_fps)
            np.save(str(inference_args.save_root / f"{save_name}_actions.npy"), actions)
            logger.info(f"Saved video to {chunk_video_name}")

    # Synchronize all processes before cleanup
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()

    # Clean up distributed resources
    video2world_cli.cleanup()
