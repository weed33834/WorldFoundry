"""
This module provides an inference wrapper for the FantasyWorld Wan2.1 model.

It handles the setup of the diffusion model, MoGe, and camera pose processors,
and offers a method to generate video sequences conditioned on an input image,
camera parameters, and text prompts.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .runtime_env import (
    DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT,
    ensure_fantasy_world_runtime,
    ensure_moge2_runtime,
    resolve_moge_pretrained,
)
from .worldfoundry_runtime import normalize_wan_num_frames, pad_camera_params_to_frames


class FantasyWorldWan21Runner:
    """
    An inference wrapper for the FantasyWorld Wan2.1 model.

    This class initializes the necessary models and components for generating
    video from an initial image, camera poses, and text prompts.
    It manages the loading of the main diffusion model, the MoGe depth model,
    and the pose processing utilities.
    """

    def __init__(
        self,
        *,
        ckpt_dir: str,
        model_ckpt: str,
        moge_path: Optional[str] = None,
        moge_pretrained: Optional[str] = None,
        sample_steps: int = 50,
        sample_guide_scale: float = 5.0,
        frames: int = 81,
        fps: int = 16,
        height: int = 336,
        width: int = 592,
        start_index: int = 16,
        device: str = "cuda",
        weight_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """
        Initializes the FantasyWorld Wan2.1 inference runner.

        Args:
            ckpt_dir (str): Directory containing the diffusion model checkpoints.
            model_ckpt (str): Path to the main FantasyWorld Wan2.1 model checkpoint.
            moge_path (Optional[str]): Path to the MoGe model. If None, uses default runtime path.
            moge_pretrained (Optional[str]): Identifier for a pre-trained MoGe model.
                                             If None, resolves to a default.
            sample_steps (int): Number of inference steps for video generation.
            sample_guide_scale (float): Classifier-free guidance scale.
            frames (int): Number of frames to generate in the video.
            fps (int): Frames per second for the generated video.
            height (int): Height of the generated video frames.
            width (int): Width of the generated video frames.
            start_index (int): Index to start cross-attention layers.
            device (str): Device to run the models on (e.g., "cuda"). Must be a CUDA device.
            weight_dtype (torch.dtype): Data type for model weights (e.g., torch.bfloat16).

        Raises:
            ValueError: If a non-CUDA device is specified or CUDA is not available.
            RuntimeError: If unexpected keys are found during model checkpoint loading.
        """
        if not str(device).startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("FantasyWorld Wan2.1 official inference requires a CUDA device.")

        # Ensure runtime environments for FantasyWorld and MoGe are set up
        ensure_fantasy_world_runtime()
        ensure_moge2_runtime(moge_path)

        # Import models and utilities dynamically after runtime environment is ensured
        from worldfoundry.base_models.three_dimensions.depth.moge.model.v2 import MoGeModel
        from worldfoundry.base_models.diffusion_model.diffsynth.utils.re10k_pose import (
            RealEstate10KPoseProcessor,
        )
        from FantasyWorld.fusion.model_wan21 import FantasyWorldFusionModel
        from worldfoundry.base_models.three_dimensions.point_clouds.vggt.vggt.variants.fantasy_world.utils.pose_enc import (
            extri_intri_to_pose_encoding,
            pose_encoding_to_extri_intri,
        )

        from . import utils as fw_utils

        # Store configuration parameters
        self.sample_steps = int(sample_steps)
        self.sample_guide_scale = float(sample_guide_scale)
        self.fps = int(fps)
        self.device = str(device)
        self.torch_dtype = weight_dtype
        self.num_frames = normalize_wan_num_frames(frames)  # Normalize number of frames to be divisible by 8
        self.height = int(height)
        self.width = int(width)
        self.start_index = int(start_index)
        self.default_negative_prompt = DEFAULT_FANTASY_WORLD_WAN21_NEGATIVE_PROMPT

        # Store references to utility functions
        self._fw_utils = fw_utils
        self._extri_intri_to_pose_encoding = extri_intri_to_pose_encoding

        # Initialize the pose processor for camera embeddings
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

        # Define paths to diffusion transformer (DiT) components and other sub-models
        dit_path = [
            [f"{ckpt_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in range(1, 8)],
            f"{ckpt_dir}/Wan2.1_VAE.pth",
            f"{ckpt_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"{ckpt_dir}/models_t5_umt5-xxl-enc-bf16.pth",
        ]

        # Configuration for VGGT (Video Grounded Global Tracking)
        vggt_cfg = {
            "enable_camera": True,
            "enable_depth": True,
            "enable_point": True,
            "enable_track": False,
            "DPT_patch_size": 16,
        }
        # Configuration for camera control module
        camera_cfg = {
            "pose_in_dim": 768,
            "plucker_fea_dim": 2048,
            "pose_inject_method": "adaln",
            "use_info": "plucker",
        }

        # Initialize the main FantasyWorld Fusion Model
        self.model = FantasyWorldFusionModel(
            start_index=self.start_index,
            use_gradient_checkpointing=True,
            cross_attention_list=list(range(24)),
            dit_path=dit_path,
            vggt_cfg=vggt_cfg,
            camera_control=True,
            camera_cfg=camera_cfg,
        )

        # Load the main model checkpoint
        ckpt = torch.load(model_ckpt, map_location="cpu")
        messages = self.model.load_state_dict(ckpt, strict=False)
        if messages.unexpected_keys:
            raise RuntimeError(f"Unexpected FantasyWorld Wan2.1 keys: {messages.unexpected_keys}")

        # Load and configure the MoGe (Monocular Geometric) model
        self.moge = MoGeModel.from_pretrained(resolve_moge_pretrained(moge_pretrained)).to(self.device).eval()
        # Move models to specified device and data type
        self.model.to(self.torch_dtype)
        self.model.to(self.device)
        self.model.pipe.device = self.device
        self.model.eval()

    def generate_video(
        self,
        *,
        image: Image.Image,
        camera_params,
        prompt: str,
        neg_prompt: Optional[str] = None,
        using_scale: bool = True,
        seed: int = 1024,
    ):
        """
        Generates a video based on an input image, camera parameters, and a text prompt.

        Args:
            image (Image.Image): The input PIL image (first frame).
            camera_params: A list or similar iterable of camera pose objects,
                           defining the camera movement for the video.
            prompt (str): The positive text prompt for guiding video generation.
            neg_prompt (Optional[str]): The negative text prompt. If None, uses the default.
            using_scale (bool): Whether to normalize the scene scale using MoGe predictions.
            seed (int): Random seed for reproducibility.

        Returns:
            Tuple[np.ndarray, Any]:
                - frames_np_processed (np.ndarray): A NumPy array of the generated video frames,
                                                    in (T, H, W, C) format, with pixel values 0-255.
                - prediction (Any): The raw prediction output from the model's generation step.
        """
        neg_prompt = self.default_negative_prompt if neg_prompt is None else neg_prompt

        with torch.no_grad():
            input_image = image.convert("RGB")
            # Pad camera parameters to match the target number of frames
            camera_params = pad_camera_params_to_frames(camera_params, self.num_frames)
            input_array = np.array(input_image)
            # Convert input image to a PyTorch tensor and normalize to [0, 1]
            input_tensor = torch.tensor(
                input_array / 255,
                dtype=torch.float32,
                device=self.device,
            ).permute(2, 0, 1)  # Change from HWC to CHW

            # Infer depth and other MoGe features from the input image
            output = self.moge.infer(input_tensor)
            moge = {k: v.cpu().contiguous() for k, v in output.items()}

            intrinsics = []
            extrinsics = []
            # Extract intrinsic and extrinsic matrices from camera parameters
            for camera in camera_params:
                intrinsics.append(self._fw_utils.get_intrinsic_matrix(camera))
                extrinsics.append(camera.w2c_mat)
            intrinsics = torch.from_numpy(np.stack(intrinsics).astype(np.float32))
            extrinsics = torch.from_numpy(np.stack(extrinsics).astype(np.float32))
            extrinsics_4x4 = extrinsics.unsqueeze(0)  # Add batch dimension

            if using_scale:
                # Use MoGe depth to normalize the scene scale based on the first frame
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

            # Convert extrinsic and intrinsic matrices to a unified pose encoding
            pose_enc = self._extri_intri_to_pose_encoding(
                extrinsics.unsqueeze(0),
                intrinsics.unsqueeze(0),
                [self.height, self.width],
                pose_encoding_type="absT_quaR_FoV",
            ).squeeze(0)
            # Generate Plucker embedding from the pose encoding for camera control
            plucker_embedding = self.pose_processor.get_plucker_embedding_direct_from_cam_params(
                pose_enc.unsqueeze(0),
                image_size=(self.height, self.width),
            ).to(self.device, self.torch_dtype)

            # Encode the input image to get CLIP features and VAE latent representations
            image_emb = self.model.pipe.encode_image(
                input_image,
                None,
                self.num_frames,
                self.height,
                self.width,
            )
            clip_feature = image_emb["clip_feature"].to(self.device, self.torch_dtype)
            y = image_emb["y"].to(self.device, self.torch_dtype)
            # Encode positive and negative prompts into context embeddings
            ctx_pos = self.model.pipe.encode_prompt(prompt or "")["context"].to(self.device, self.torch_dtype)
            ctx_neg = self.model.pipe.encode_prompt(neg_prompt or "")["context"].to(self.device, self.torch_dtype)

        # Set up autocasting context for mixed-precision inference if using CUDA
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with torch.no_grad(), autocast_ctx:
            # Generate the latent video sequence
            latent_video, prediction = self.model.generate_video(
                context_pos=ctx_pos,
                context_neg=ctx_neg,
                clip_feature=clip_feature,
                y=y,
                height=self.height,
                width=self.width,
                num_inference_steps=self.sample_steps,
                num_frames=self.num_frames,
                image_path=None,
                plucker_embedding=plucker_embedding,
                seed=seed,
            )
            # Decode the latent video into pixel frames using the VAE
            frames = self.model.pipe.vae.decode(
                latent_video,
                device=self.device,
                tiled=True,  # Use tiled decoding for potentially larger videos
                tile_size=(30, 52),
                tile_stride=(15, 26),
            )
            # Post-process frames: permute dimensions, normalize, scale to 0-255, and convert to NumPy
            video = frames.squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu()
            video = (video + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
            video = (video * 255.0).clamp(0, 255)  # Scale to [0, 255]
            frames_np_processed = video.numpy().astype(np.uint8)

        return frames_np_processed, prediction