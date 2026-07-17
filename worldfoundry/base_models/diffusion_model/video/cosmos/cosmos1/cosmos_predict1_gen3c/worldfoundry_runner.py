"""Module for base_models -> diffusion_model -> video -> cosmos -> cosmos1 -> cosmos_predict1_gen3c -> worldfoundry_runner.py functionality."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path


def _restrict_visible_cuda_devices_for_torchrun() -> None:
    """Helper function to restrict visible cuda devices for torchrun.

    Returns:
        The return value.
    """
    if os.environ.get("WORLDFOUNDRY_GEN3C_PER_RANK_CUDA_VISIBLE", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    local_rank_text = os.environ.get("LOCAL_RANK")
    if local_rank_text is None:
        return
    try:
        local_rank = int(local_rank_text)
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("WORLD_SIZE") or "1")
    except ValueError:
        return
    if local_world_size <= 1:
        return

    visible_devices = [item.strip() for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item.strip()]
    if visible_devices and local_rank < len(visible_devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    os.environ["WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE"] = "0"


_restrict_visible_cuda_devices_for_torchrun()


def _stagger_torchrun_imports() -> None:
    """Stagger heavy imports across ranks on shared filesystems."""
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0") or "0")
        seconds = float(os.environ.get("WORLDFOUNDRY_GEN3C_IMPORT_STAGGER_SECONDS", "0") or "0")
    except ValueError:
        return
    if local_rank > 0 and seconds > 0:
        time.sleep(local_rank * seconds)


_stagger_torchrun_imports()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.enable_grad(False)


def parse_args() -> argparse.Namespace:
    """Parse args.

    Returns:
        The return value.
    """
    parser = argparse.ArgumentParser(description="WorldFoundry GEN3C single-image runner.")
    parser.add_argument("--checkpoint_dir", required=True, type=str)
    parser.add_argument("--input_image_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--scene_name", default="gen3c_scene", type=str)
    parser.add_argument("--video_save_name", default="video", type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--negative_prompt", default="", type=str)
    parser.add_argument("--trajectory", default="left", type=str)
    parser.add_argument(
        "--camera_rotation",
        default="center_facing",
        choices=["center_facing", "no_rotation", "trajectory_aligned"],
        type=str,
    )
    parser.add_argument("--movement_distance", default=0.3, type=float)
    parser.add_argument("--guidance", default=1.0, type=float)
    parser.add_argument("--num_steps", default=35, type=int)
    parser.add_argument("--num_video_frames", default=121, type=int)
    parser.add_argument("--fps", default=24, type=int)
    parser.add_argument("--height", default=704, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--noise_aug_strength", default=0.0, type=float)
    parser.add_argument("--save_buffer", action="store_true")
    parser.add_argument("--filter_points_threshold", default=0.05, type=float)
    parser.add_argument("--foreground_masking", action="store_true")
    parser.add_argument("--disable_prompt_upsampler", action="store_true")
    parser.add_argument("--offload_diffusion_transformer", action="store_true")
    parser.add_argument("--offload_tokenizer", action="store_true")
    parser.add_argument("--offload_text_encoder_model", action="store_true")
    parser.add_argument("--offload_prompt_upsampler", action="store_true")
    parser.add_argument("--offload_guardrail_models", action="store_true")
    parser.add_argument("--disable_guardrail", action="store_true")
    parser.add_argument("--disable_prompt_encoder", action="store_true")
    parser.add_argument("--moge_pretrained", required=True, type=str)
    return parser.parse_args()


def _alias_package(alias: str, target: str) -> None:
    """Helper function to alias package.

    Args:
        alias: The alias.
        target: The target.

    Returns:
        The return value.
    """
    module = importlib.import_module(target)
    sys.modules[alias] = module


def _prepend_sys_path(path: Path) -> None:
    """Helper function to prepend sys path.

    Args:
        path: The path.

    Returns:
        The return value.
    """
    path_text = str(path)
    if path_text in sys.path:
        sys.path.remove(path_text)
    sys.path.insert(0, path_text)


def _ensure_gen3c_runtime_package() -> str:
    """Helper function to ensure gen3c runtime package.

    Returns:
        The return value.
    """
    from .runtime_env import official_gen3c_repo_root, packaged_gen3c_cosmos_root

    if "cosmos_predict1" in sys.modules:
        module = sys.modules["cosmos_predict1"]
        return str(getattr(module, "__file__", "already-imported"))

    official_root = official_gen3c_repo_root()
    if official_root is not None:
        _prepend_sys_path(official_root)
        return str(official_root)

    packaged_root = packaged_gen3c_cosmos_root()
    _prepend_sys_path(packaged_root)
    if importlib.util.find_spec("cosmos_predict1") is None:
        _alias_package(
            "cosmos_predict1",
            "worldfoundry.base_models.diffusion_model.video.cosmos.cosmos1.cosmos_predict1_gen3c.cosmos_predict1",
        )
    return str(packaged_root)


def _ensure_utils3d_alias() -> None:
    """Helper function to ensure utils3d alias.

    Returns:
        The return value.
    """
    if "utils3d" in sys.modules or importlib.util.find_spec("utils3d") is not None:
        return

    from worldfoundry.base_models.three_dimensions.general_3d.eastern_journalist import (
        utils3d as vendored_utils3d,
    )

    sys.modules["utils3d"] = vendored_utils3d


def _validate_num_frames(num_video_frames: int) -> None:
    """Helper function to validate num frames.

    Args:
        num_video_frames: The num video frames.

    Returns:
        The return value.
    """
    if num_video_frames is None or (num_video_frames - 1) % 120 != 0:
        raise ValueError("num_video_frames must be 121, 241, 361, ... (N * 120 + 1).")


def _select_device(num_gpus: int) -> torch.device:
    """Helper function to select device.

    Args:
        num_gpus: The num gpus.

    Returns:
        The return value.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    local_device = os.environ.get("WORLDFOUNDRY_GEN3C_LOCAL_CUDA_DEVICE")
    if local_device is not None:
        torch.cuda.set_device(int(local_device))
        return torch.device(f"cuda:{local_device}")
    if num_gpus > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda")


def _enable_context_parallel(pipeline, process_group) -> None:
    """Helper function to enable context parallel.

    Args:
        pipeline: The pipeline.
        process_group: The process group.

    Returns:
        The return value.
    """
    model = getattr(pipeline, "model", None)
    net = getattr(model, "net", None)
    if net is not None:
        net.enable_context_parallel(process_group)
    else:
        setattr(pipeline, "_worldfoundry_context_parallel_group", process_group)


def _initialize_context_parallel(num_gpus: int):
    """Helper function to initialize context parallel.

    Args:
        num_gpus: The num gpus.
    """
    if num_gpus <= 1:
        return None, None

    try:
        from cosmos_predict1.utils import distributed

        from worldfoundry.core.distributed.megatron_compat import parallel_state
    except Exception:
        from worldfoundry.core.distributed import torch_process_group as distributed
        from worldfoundry.core.distributed.megatron_compat import parallel_state

    distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=num_gpus)
    return parallel_state.get_context_parallel_group(), parallel_state


def _rank0_print(message: str) -> None:
    """Helper function to rank0 print.

    Args:
        message: The message.

    Returns:
        The return value.
    """
    if os.environ.get("RANK", "0") == "0":
        print(message, flush=True)


def _is_rank0() -> bool:
    """Helper function to is rank0.

    Returns:
        The return value.
    """
    return os.environ.get("RANK", "0") == "0"


def _offload_cuda_module(module: torch.nn.Module, device: torch.device) -> None:
    """Helper function to offload cuda module.

    Args:
        module: The module.
        device: The device.

    Returns:
        The return value.
    """
    if device.type != "cuda":
        return
    module.to("cpu")
    torch.cuda.empty_cache()


def _load_cuda_module(module: torch.nn.Module, device: torch.device) -> None:
    """Helper function to load cuda module.

    Args:
        module: The module.
        device: The device.

    Returns:
        The return value.
    """
    if device.type != "cuda":
        return
    module.to(device)


def _predict_moge_depth(current_image_path: str, target_h: int, target_w: int, device, moge_model):
    """Helper function to predict moge depth.

    Args:
        current_image_path: The current image path.
        target_h: The target h.
        target_w: The target w.
        device: The device.
        moge_model: The moge model.
    """
    input_image_bgr = cv2.imread(current_image_path)
    if input_image_bgr is None:
        raise FileNotFoundError(f"Input image not found: {current_image_path}")
    input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)

    depth_pred_h, depth_pred_w = 720, 1280
    input_image_for_depth_resized = cv2.resize(input_image_rgb, (depth_pred_w, depth_pred_h))
    input_image_for_depth_tensor_chw = torch.tensor(
        input_image_for_depth_resized / 255.0,
        dtype=torch.float32,
        device=device,
    ).permute(2, 0, 1)
    moge_output_full = moge_model.infer(input_image_for_depth_tensor_chw)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_intrinsics_33_full_normalized = moge_output_full["intrinsics"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_hw_full = torch.where(
        moge_mask_hw_full == 0,
        torch.tensor(1000.0, device=moge_depth_hw_full.device),
        moge_depth_hw_full,
    )
    moge_intrinsics_33_full_pixel = moge_intrinsics_33_full_normalized.clone()
    moge_intrinsics_33_full_pixel[0, 0] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 1] *= depth_pred_h
    moge_intrinsics_33_full_pixel[0, 2] *= depth_pred_w
    moge_intrinsics_33_full_pixel[1, 2] *= depth_pred_h

    height_scale_factor = target_h / depth_pred_h
    width_scale_factor = target_w / depth_pred_w

    moge_depth_hw = (
        F.interpolate(
            moge_depth_hw_full.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    moge_mask_hw = (
        F.interpolate(
            moge_mask_hw_full.unsqueeze(0).unsqueeze(0).to(torch.float32),
            size=(target_h, target_w),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .to(torch.bool)
    )
    input_image_tensor_chw_target_res = F.interpolate(
        input_image_for_depth_tensor_chw.unsqueeze(0),
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    moge_image_b1chw_float = input_image_tensor_chw_target_res.unsqueeze(0).unsqueeze(1) * 2 - 1
    moge_intrinsics_33 = moge_intrinsics_33_full_pixel.clone()
    moge_intrinsics_33[1, 1] *= height_scale_factor
    moge_intrinsics_33[1, 2] *= height_scale_factor
    moge_intrinsics_33[0, 0] *= width_scale_factor
    moge_intrinsics_33[0, 2] *= width_scale_factor

    moge_depth_b11hw = moge_depth_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_depth_b11hw = torch.nan_to_num(moge_depth_b11hw, nan=1e4)
    moge_depth_b11hw = torch.clamp(moge_depth_b11hw, min=0, max=1e4)
    moge_mask_b11hw = moge_mask_hw.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    moge_intrinsics_b133 = moge_intrinsics_33.unsqueeze(0).unsqueeze(0)
    initial_w2c_44 = torch.eye(4, dtype=torch.float32, device=device)
    moge_initial_w2c_b144 = initial_w2c_44.unsqueeze(0).unsqueeze(0)

    return (
        moge_image_b1chw_float,
        moge_depth_b11hw,
        moge_mask_b11hw,
        moge_initial_w2c_b144,
        moge_intrinsics_b133,
    )


def _predict_moge_depth_from_tensor(image_tensor_chw_0_1: torch.Tensor, moge_model):
    """Helper function to predict moge depth from tensor.

    Args:
        image_tensor_chw_0_1: The image tensor chw 0 1.
        moge_model: The moge model.
    """
    moge_output_full = moge_model.infer(image_tensor_chw_0_1)
    moge_depth_hw_full = moge_output_full["depth"]
    moge_mask_hw_full = moge_output_full["mask"]

    moge_depth_11hw = moge_depth_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.nan_to_num(moge_depth_11hw, nan=1e4)
    moge_depth_11hw = torch.clamp(moge_depth_11hw, min=0, max=1e4)
    moge_mask_11hw = moge_mask_hw_full.unsqueeze(0).unsqueeze(0)
    moge_depth_11hw = torch.where(
        moge_mask_11hw == 0,
        torch.tensor(1000.0, device=moge_depth_11hw.device),
        moge_depth_11hw,
    )
    return moge_depth_11hw, moge_mask_11hw


def main() -> None:
    """Main.

    Returns:
        The return value.
    """
    args = parse_args()
    _validate_num_frames(args.num_video_frames)
    runtime_root = _ensure_gen3c_runtime_package()
    _ensure_utils3d_alias()

    import cosmos_predict1
    from cosmos_predict1.diffusion.inference.cache_3d import Cache3D_Buffer
    from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
    from cosmos_predict1.diffusion.inference.gen3c_pipeline import Gen3cPipeline
    from cosmos_predict1.utils import misc

    from worldfoundry.base_models.diffusion_model.video.cosmos.shared.io import save_video
    from worldfoundry.base_models.three_dimensions.depth.moge.model.v1 import MoGeModel

    _rank0_print(
        "WorldFoundry GEN3C runtime: "
        f"root={runtime_root}; cosmos_predict1={getattr(cosmos_predict1, '__file__', None)}; "
        f"per_rank_cuda_visible={os.environ.get('WORLDFOUNDRY_GEN3C_PER_RANK_CUDA_VISIBLE', '0')}"
    )
    misc.set_random_seed(args.seed)
    process_group, parallel_state = _initialize_context_parallel(args.num_gpus)
    device = _select_device(args.num_gpus)

    try:
        pipeline = Gen3cPipeline(
            inference_type="video2world",
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_name="Gen3C-Cosmos-7B",
            prompt_upsampler_dir="Pixtral-12B",
            enable_prompt_upsampler=not args.disable_prompt_upsampler,
            offload_network=args.offload_diffusion_transformer,
            offload_tokenizer=args.offload_tokenizer,
            offload_text_encoder_model=args.offload_text_encoder_model,
            offload_prompt_upsampler=args.offload_prompt_upsampler,
            offload_guardrail_models=args.offload_guardrail_models,
            disable_guardrail=args.disable_guardrail,
            disable_prompt_encoder=args.disable_prompt_encoder,
            guidance=args.guidance,
            num_steps=args.num_steps,
            height=args.height,
            width=args.width,
            fps=args.fps,
            num_video_frames=121,
            seed=args.seed,
        )

        frame_buffer_max = pipeline.model.frame_buffer_max
        generator = torch.Generator(device=device).manual_seed(args.seed)
        sample_n_frames = pipeline.model.chunk_size
        moge_model = MoGeModel.from_pretrained(args.moge_pretrained).to(device).eval()

        if process_group is not None:
            _enable_context_parallel(pipeline, process_group)

        (
            moge_image_b1chw_float,
            moge_depth_b11hw,
            _moge_mask_b11hw,
            moge_initial_w2c_b144,
            moge_intrinsics_b133,
        ) = _predict_moge_depth(
            args.input_image_path,
            args.height,
            args.width,
            device,
            moge_model,
        )
        _offload_cuda_module(moge_model, device)

        cache = Cache3D_Buffer(
            frame_buffer_max=frame_buffer_max,
            generator=generator,
            noise_aug_strength=args.noise_aug_strength,
            input_image=moge_image_b1chw_float[:, 0].clone(),
            input_depth=moge_depth_b11hw[:, 0],
            input_w2c=moge_initial_w2c_b144[:, 0],
            input_intrinsics=moge_intrinsics_b133[:, 0],
            filter_points_threshold=args.filter_points_threshold,
            foreground_masking=args.foreground_masking,
        )

        initial_cam_w2c_for_traj = moge_initial_w2c_b144[0, 0]
        initial_cam_intrinsics_for_traj = moge_intrinsics_b133[0, 0]
        generated_w2cs, generated_intrinsics = generate_camera_trajectory(
            trajectory_type=args.trajectory,
            initial_w2c=initial_cam_w2c_for_traj,
            initial_intrinsics=initial_cam_intrinsics_for_traj,
            num_frames=args.num_video_frames,
            movement_distance=args.movement_distance,
            camera_rotation=args.camera_rotation,
            center_depth=1.0,
            device=device.type,
        )

        rendered_warp_images, rendered_warp_masks = cache.render_cache(
            generated_w2cs[:, 0:sample_n_frames],
            generated_intrinsics[:, 0:sample_n_frames],
        )

        all_rendered_warps = []
        if args.save_buffer:
            all_rendered_warps.append(rendered_warp_images.clone().cpu())

        generated_output = pipeline.generate(
            prompt=args.prompt,
            image_path=args.input_image_path,
            negative_prompt=args.negative_prompt,
            rendered_warp_images=rendered_warp_images,
            rendered_warp_masks=rendered_warp_masks,
        )
        if generated_output is None:
            raise RuntimeError("GEN3C guardrail blocked generation.")
        video, _ = generated_output

        num_ar_iterations = (generated_w2cs.shape[1] - 1) // (sample_n_frames - 1)
        for num_iter in range(1, num_ar_iterations):
            start_frame_idx = num_iter * (sample_n_frames - 1)
            end_frame_idx = start_frame_idx + sample_n_frames

            last_frame_hwc_0_255 = torch.tensor(video[-1], device=device)
            pred_image_for_depth_chw_0_1 = last_frame_hwc_0_255.permute(2, 0, 1) / 255.0
            _load_cuda_module(moge_model, device)
            pred_depth, _pred_mask = _predict_moge_depth_from_tensor(
                pred_image_for_depth_chw_0_1,
                moge_model,
            )
            _offload_cuda_module(moge_model, device)

            cache.update_cache(
                new_image=pred_image_for_depth_chw_0_1.unsqueeze(0) * 2 - 1,
                new_depth=pred_depth,
                new_w2c=generated_w2cs[:, start_frame_idx],
                new_intrinsics=generated_intrinsics[:, start_frame_idx],
            )
            current_segment_w2cs = generated_w2cs[:, start_frame_idx:end_frame_idx]
            current_segment_intrinsics = generated_intrinsics[:, start_frame_idx:end_frame_idx]
            rendered_warp_images, rendered_warp_masks = cache.render_cache(
                current_segment_w2cs,
                current_segment_intrinsics,
            )

            if args.save_buffer:
                all_rendered_warps.append(rendered_warp_images[:, 1:].clone().cpu())

            pred_image_for_depth_bcthw_minus1_1 = pred_image_for_depth_chw_0_1.unsqueeze(0).unsqueeze(2) * 2 - 1
            generated_output = pipeline.generate(
                prompt=args.prompt,
                image_path=pred_image_for_depth_bcthw_minus1_1,
                negative_prompt=args.negative_prompt,
                rendered_warp_images=rendered_warp_images,
                rendered_warp_masks=rendered_warp_masks,
            )
            if generated_output is None:
                raise RuntimeError("GEN3C guardrail blocked autoregressive generation.")
            video_new, _ = generated_output
            video = np.concatenate([video, video_new[1:]], axis=0)

        final_video_to_save = video
        final_width = args.width
        if args.save_buffer and all_rendered_warps:
            squeezed_warps = [tensor.squeeze(0) for tensor in all_rendered_warps]
            if squeezed_warps:
                n_max = max(tensor.shape[1] for tensor in squeezed_warps)
                padded_tensors = []
                for sq_tensor in squeezed_warps:
                    padding_needed_dim1 = n_max - sq_tensor.shape[1]
                    pad_spec = (0, 0, 0, 0, 0, 0, 0, padding_needed_dim1, 0, 0)
                    padded_tensors.append(F.pad(sq_tensor, pad_spec, mode="constant", value=-1.0))

                full_rendered_warp_tensor = torch.cat(padded_tensors, dim=0)
                t_total, _, c_dim, h_dim, w_dim = full_rendered_warp_tensor.shape
                buffer_video_tchnw = full_rendered_warp_tensor.permute(0, 2, 3, 1, 4)
                buffer_video_tchw_stacked = buffer_video_tchnw.contiguous().view(
                    t_total,
                    c_dim,
                    h_dim,
                    n_max * w_dim,
                )
                buffer_video_tchw_stacked = (buffer_video_tchw_stacked * 0.5 + 0.5) * 255.0
                buffer_numpy_tchw_stacked = buffer_video_tchw_stacked.cpu().numpy().astype(np.uint8)
                buffer_numpy_thwc = np.transpose(buffer_numpy_tchw_stacked, (0, 2, 3, 1))
                final_video_to_save = np.concatenate([buffer_numpy_thwc, final_video_to_save], axis=2)
                final_width = args.width * (1 + n_max)

        if not _is_rank0():
            return

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        video_save_path = output_dir / f"{args.video_save_name}.mp4"
        save_video(
            video=final_video_to_save,
            fps=args.fps,
            H=args.height,
            W=final_width,
            video_save_quality=5,
            video_save_path=str(video_save_path),
        )

        result = {
            "generated_video_path": str(video_save_path.resolve()),
            "output_dir": str(output_dir),
            "scene_name": args.scene_name,
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "trajectory": args.trajectory,
            "camera_rotation": args.camera_rotation,
            "movement_distance": args.movement_distance,
            "fps": args.fps,
            "height": args.height,
            "width": final_width,
            "num_video_frames": int(video.shape[0]),
        }
        with (output_dir / "result.json").open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
    finally:
        if parallel_state is not None:
            parallel_state.destroy_model_parallel()
            import torch.distributed as dist

            dist.destroy_process_group()


if __name__ == "__main__":
    main()
