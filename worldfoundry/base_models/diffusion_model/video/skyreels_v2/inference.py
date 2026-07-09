"""Module for base_models -> diffusion_model -> video -> skyreels_v2 -> inference.py functionality."""

from __future__ import annotations

import argparse
import random
import time
from contextlib import nullcontext
from pathlib import Path


NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly "
    "drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"
)


def _optional_path(value: str | None) -> str | None:
    """Helper function to optional path.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _infer_resolution(model_id: str, explicit: str | None) -> str:
    """Helper function to infer resolution.

    Args:
        model_id: The model id.
        explicit: The explicit.

    Returns:
        The return value.
    """
    if explicit:
        return explicit
    name = model_id.upper()
    if "720P" in name:
        return "720P"
    return "540P"


def _resolution_size(resolution: str) -> tuple[int, int]:
    """Helper function to resolution size.

    Args:
        resolution: The resolution.

    Returns:
        The return value.
    """
    if resolution == "540P":
        return 544, 960
    if resolution == "720P":
        return 720, 1280
    raise ValueError(f"Invalid resolution: {resolution}")


def _infer_task(model_id: str, image_path: str | None, video_path: str | None, task: str) -> str:
    """Helper function to infer task.

    Args:
        model_id: The model id.
        image_path: The image path.
        video_path: The video path.
        task: The task.

    Returns:
        The return value.
    """
    if task != "auto":
        return task
    name = model_id.upper()
    if "DF" in name or video_path:
        return "df"
    if image_path or "I2V" in name:
        return "i2v"
    return "t2v"


def _seed(value: int | None) -> int:
    """Helper function to seed.

    Args:
        value: The value.

    Returns:
        The return value.
    """
    if value is not None:
        return int(value)
    random.seed(time.time())
    return int(random.randrange(4294967294))


def _autocast_context(torch_module, device: str, dtype):
    """Helper function to autocast context.

    Args:
        torch_module: The torch module.
        device: The device.
        dtype: The dtype.
    """
    if str(device).startswith("cuda"):
        return torch_module.cuda.amp.autocast(dtype=dtype)
    return nullcontext()


def _write_video(frames, save_path: str | Path, fps: int) -> None:
    """Helper function to write video.

    Args:
        frames: The frames.
        save_path: The save path.
        fps: The fps.

    Returns:
        The return value.
    """
    import imageio

    output = Path(save_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output), frames, fps=fps, quality=8, output_params=["-loglevel", "error"])
    print(f"saved video: {output}")


def _initialize_usp(prompt_enhancer: bool) -> int:
    """Helper function to initialize usp.

    Args:
        prompt_enhancer: The prompt enhancer.

    Returns:
        The return value.
    """
    if prompt_enhancer:
        raise ValueError(
            "--prompt_enhancer is not allowed with --use_usp. Run prompt enhancement before distributed inference."
        )
    from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
    import torch
    import torch.distributed as dist

    dist.init_process_group("nccl")
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    init_distributed_environment(rank=local_rank, world_size=dist.get_world_size())
    initialize_model_parallel(
        sequence_parallel_degree=dist.get_world_size(),
        ring_degree=1,
        ulysses_degree=dist.get_world_size(),
    )
    return local_rank


def _enhance_prompt(prompt: str) -> str:
    """Helper function to enhance prompt.

    Args:
        prompt: The prompt.

    Returns:
        The return value.
    """
    from skyreels_v2_infer.pipelines import PromptEnhancer

    print("init prompt enhancer")
    prompt_enhancer = PromptEnhancer()
    enhanced = prompt_enhancer(prompt)
    print(f"enhanced prompt: {enhanced}")
    return enhanced


def _run_standard(args: argparse.Namespace, model_id: str, task: str, seed: int) -> None:
    """Helper function to run standard.

    Args:
        args: The args.
        model_id: The model id.
        task: The task.
        seed: The seed.

    Returns:
        The return value.
    """
    import gc

    import torch
    from diffusers.utils import load_image
    from skyreels_v2_infer.pipelines import Image2VideoPipeline, Text2VideoPipeline, resizecrop

    height, width = _resolution_size(args.resolution)
    image = load_image(args.image).convert("RGB") if args.image else None
    prompt = _enhance_prompt(args.prompt) if args.prompt_enhancer and image is None else args.prompt

    if task == "i2v" and image is None:
        raise ValueError("SkyReels-V2 I2V inference requires --image.")

    if image is None:
        print("init text2video pipeline")
        pipe = Text2VideoPipeline(
            model_path=model_id,
            dit_path=model_id,
            device=args.device,
            use_usp=args.use_usp,
            offload=args.offload,
        )
    else:
        print("init img2video pipeline")
        pipe = Image2VideoPipeline(
            model_path=model_id,
            dit_path=model_id,
            device=args.device,
            use_usp=args.use_usp,
            offload=args.offload,
        )
        image_width, image_height = image.size
        if image_height > image_width:
            height, width = width, height
        image = resizecrop(image, height, width)

    if args.teacache:
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=args.inference_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=model_id,
        )

    generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
    kwargs = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "num_frames": args.num_frames,
        "num_inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "fps": args.fps,
        "generator": torch.Generator(device=generator_device).manual_seed(seed),
        "height": height,
        "width": width,
    }
    if image is not None:
        kwargs["image"] = image.convert("RGB")

    with _autocast_context(torch, args.device, pipe.transformer.dtype), torch.no_grad():
        print(f"infer task:{task} kwargs:{kwargs}")
        frames = pipe(**kwargs)[0]

    _write_video(frames, args.save_path, args.fps)
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()


def _run_diffusion_forcing(args: argparse.Namespace, model_id: str, seed: int) -> None:
    """Helper function to run diffusion forcing.

    Args:
        args: The args.
        model_id: The model id.
        seed: The seed.

    Returns:
        The return value.
    """
    import torch
    from diffusers.utils import load_image
    from skyreels_v2_infer import DiffusionForcingPipeline
    from skyreels_v2_infer.pipelines.image2video_pipeline import resizecrop

    height, width = _resolution_size(args.resolution)
    prompt = _enhance_prompt(args.prompt) if args.prompt_enhancer and not args.image else args.prompt

    if args.num_frames > args.base_num_frames and args.overlap_history is None:
        raise ValueError(
            'Specify "--overlap_history" for long-video generation when num_frames exceeds base_num_frames.'
        )
    if args.addnoise_condition > 60:
        print('warning: "addnoise_condition" is usually recommended to be 20 or lower.')

    pipe = DiffusionForcingPipeline(
        model_id,
        dit_path=model_id,
        device=torch.device(args.device),
        weight_dtype=torch.bfloat16,
        use_usp=args.use_usp,
        offload=args.offload,
    )
    if args.causal_attention:
        pipe.transformer.set_ar_attention(args.causal_block_size)
    if args.teacache:
        if args.ar_step > 0:
            num_steps = args.inference_steps + (
                ((args.base_num_frames - 1) // 4 + 1) // args.causal_block_size - 1
            ) * args.ar_step
        else:
            num_steps = args.inference_steps
        pipe.transformer.initialize_teacache(
            enable_teacache=True,
            num_steps=num_steps,
            teacache_thresh=args.teacache_thresh,
            use_ret_steps=args.use_ret_steps,
            ckpt_dir=model_id,
        )

    generator_device = args.device if str(args.device).startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    video_path = _optional_path(args.video_path)
    if video_path:
        frames = pipe.extend_video(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            prefix_video_path=video_path,
            height=height,
            width=width,
            num_frames=args.num_frames,
            num_inference_steps=args.inference_steps,
            shift=args.shift,
            guidance_scale=args.guidance_scale,
            generator=generator,
            overlap_history=args.overlap_history,
            addnoise_condition=args.addnoise_condition,
            base_num_frames=args.base_num_frames,
            ar_step=args.ar_step,
            causal_block_size=args.causal_block_size,
            fps=args.fps,
        )[0]
    else:
        image = load_image(args.image) if args.image else None
        end_image = load_image(args.end_image) if args.end_image else None
        if image is not None:
            image_width, image_height = image.size
            if image_height > image_width:
                height, width = width, height
            image = resizecrop(image, height, width).convert("RGB")
            if end_image is not None:
                end_image = resizecrop(end_image, height, width).convert("RGB")

        with _autocast_context(torch, args.device, pipe.transformer.dtype), torch.no_grad():
            frames = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                image=image,
                end_image=end_image,
                height=height,
                width=width,
                num_frames=args.num_frames,
                num_inference_steps=args.inference_steps,
                shift=args.shift,
                guidance_scale=args.guidance_scale,
                generator=generator,
                overlap_history=args.overlap_history,
                addnoise_condition=args.addnoise_condition,
                base_num_frames=args.base_num_frames,
                ar_step=args.ar_step,
                causal_block_size=args.causal_block_size,
                fps=args.fps,
            )[0]

    _write_video(frames, args.save_path, args.fps)


def build_parser() -> argparse.ArgumentParser:
    """Build parser.

    Returns:
        The return value.
    """
    parser = argparse.ArgumentParser(description="WorldFoundry in-tree launcher for SkyReels-V2 official inference.")
    parser.add_argument("--model_id", type=str, default="Skywork/SkyReels-V2-DF-1.3B-540P")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_path", "--save-path", "--output_path", "--output-path", dest="save_path", required=True)
    parser.add_argument("--task", choices=["auto", "t2v", "i2v", "df"], default="auto")
    parser.add_argument("--resolution", type=str, choices=["540P", "720P"], default=None)
    parser.add_argument("--num_frames", type=int, default=97)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--end_image", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--ar_step", type=int, default=0)
    parser.add_argument("--causal_attention", action="store_true")
    parser.add_argument("--causal_block_size", type=int, default=1)
    parser.add_argument("--base_num_frames", type=int, default=97)
    parser.add_argument("--overlap_history", type=int, default=None)
    parser.add_argument("--addnoise_condition", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--shift", type=float, default=8.0)
    parser.add_argument("--inference_steps", type=int, default=30)
    parser.add_argument("--use_usp", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt_enhancer", action="store_true")
    parser.add_argument("--teacache", action="store_true")
    parser.add_argument("--teacache_thresh", type=float, default=0.2)
    parser.add_argument("--use_ret_steps", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Main.

    Args:
        argv: The argv.

    Returns:
        The return value.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    args.image = _optional_path(args.image)
    args.end_image = _optional_path(args.end_image)
    args.video_path = _optional_path(args.video_path)
    args.resolution = _infer_resolution(args.model_id, args.resolution)
    if args.use_usp:
        if args.seed is None:
            raise ValueError("USP mode requires an explicit --seed")
        _initialize_usp(args.prompt_enhancer)
    seed = _seed(args.seed)

    from skyreels_v2_infer.modules import download_model

    model_id = download_model(args.model_id)
    print("model_id:", model_id)
    task = _infer_task(model_id, args.image, args.video_path, args.task)
    if task == "df":
        _run_diffusion_forcing(args, model_id, seed)
    else:
        _run_standard(args, model_id, task, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
