"""Video read/write primitives shared by inference and evaluation code."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

from .media import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .storage import local_path_for_uri, parse_uri_scheme, uri_to_local_path, write_binary_uri


def extract_frames_from_video_url(video_url: str):
    """Decode a remote video URL and return RGB PIL frames."""

    from PIL import Image

    frames = load_video_frames(video_url)
    return [Image.fromarray(frame[..., :3]) for frame in frames]


def resize_video_tensor_to_resolution(video_tensor, resolution: Iterable[int]):
    """Resize and center-crop a ``[T,C,H,W]`` tensor to ``(target_h, target_w)``."""

    import torchvision

    target_h, target_w = [int(value) for value in resolution]
    orig_h, orig_w = int(video_tensor.shape[2]), int(video_tensor.shape[3])
    scaling_ratio = max(target_w / orig_w, target_h / orig_h)
    resizing_shape = (int(math.ceil(scaling_ratio * orig_h)), int(math.ceil(scaling_ratio * orig_w)))
    resized = torchvision.transforms.functional.resize(video_tensor, resizing_shape)
    return torchvision.transforms.functional.center_crop(resized, [target_h, target_w])


def read_image_as_video_tensor(
    image_path: str | Path,
    resolution: Iterable[int],
    num_video_frames: int,
    *,
    resize: bool = True,
):
    """Load an image and materialize a ``[1,C,T,H,W]`` uint8 conditioning video tensor."""

    import torch
    import torchvision
    from PIL import Image

    path = Path(image_path)
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Invalid image extension: {path.suffix}")
    if num_video_frames < 1:
        raise ValueError("num_video_frames must be at least 1")

    image = Image.open(path).convert("RGB")
    image_tensor = torchvision.transforms.functional.to_tensor(image)
    first_frame = image_tensor.unsqueeze(0)
    zero_tail = torch.zeros_like(first_frame).repeat(num_video_frames - 1, 1, 1, 1)
    video = torch.cat([first_frame, zero_tail], dim=0)
    video = (video * 255.0).to(torch.uint8)
    if resize:
        video = resize_video_tensor_to_resolution(video, resolution)
    return video.unsqueeze(0).permute(0, 2, 1, 3, 4)


def video_tensor_to_uint8_frames(video_tensor, *, value_range: str | tuple[float, float] = "auto") -> "object":
    """Convert a normalized torch video tensor to uint8 THWC frames.

    ``value_range="auto"`` treats tensors with negative values as ``[-1, 1]``
    and non-negative floating tensors as ``[0, 1]``.
    """

    import torch

    if not torch.is_tensor(video_tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(video_tensor)}")
    tensor = video_tensor.detach().cpu().float()
    if tensor.ndim == 5:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected batch size 1 for 5D video tensor, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 4:
        raise ValueError(f"Expected video tensor shape [C,T,H,W], got {tuple(tensor.shape)}")
    if tensor.shape[-1] in {1, 3, 4}:
        video = tensor
    elif tensor.shape[0] in {1, 3, 4}:
        video = tensor.permute(1, 2, 3, 0)
    elif tensor.shape[1] in {1, 3, 4}:
        video = tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Unable to infer channel layout for video tensor shape {tuple(tensor.shape)}")
    if value_range == "auto":
        low, high = (-1.0, 1.0) if float(video.min()) < 0.0 else (0.0, 1.0)
    elif value_range == "-1,1":
        low, high = -1.0, 1.0
    elif value_range == "0,1":
        low, high = 0.0, 1.0
    else:
        low, high = value_range
    if high <= low:
        raise ValueError("value_range high bound must be larger than low bound.")
    video = ((video.clamp(float(low), float(high)) - float(low)) * (255.0 / (float(high) - float(low)))).to(torch.uint8)
    return video.numpy()


def coerce_video_frames(video_input):
    """Normalize common video inputs into a uint8 THWC numpy array."""

    import os

    import numpy as np
    import torch
    from PIL import Image

    if isinstance(video_input, (str, os.PathLike)):
        return load_video_frames(str(video_input))

    if torch.is_tensor(video_input):
        return video_tensor_to_uint8_frames(video_input)

    if isinstance(video_input, np.ndarray):
        video = video_input
    elif isinstance(video_input, (list, tuple)):
        frames = []
        for frame in video_input:
            if torch.is_tensor(frame):
                tensor = frame.detach().cpu()
                if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
                    tensor = tensor.permute(1, 2, 0)
                frame = tensor.numpy()
            elif isinstance(frame, Image.Image):
                frame = np.asarray(frame.convert("RGB"))
            else:
                frame = np.asarray(frame)
            frames.append(frame)
        video = np.stack(frames, axis=0)
    else:
        raise TypeError(f"Unsupported video input type: {type(video_input)}")

    if video.ndim != 4:
        raise ValueError(f"Expected video with 4 dimensions, got shape {tuple(video.shape)}")

    if video.shape[-1] in {1, 3, 4}:
        converted = video
    elif video.shape[1] in {1, 3, 4}:
        converted = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unable to infer channel layout for video shape {tuple(video.shape)}")

    if converted.dtype != np.uint8:
        if np.issubdtype(converted.dtype, np.floating):
            if converted.max() <= 1.0:
                converted = np.clip(converted * 255.0, 0, 255)
            else:
                converted = np.clip(converted, 0, 255)
        else:
            converted = np.clip(converted, 0, 255)
        converted = converted.astype(np.uint8)
    return converted


def load_video_frames(video_path: str | Path):
    """Decode a local or remote video into a uint8 THWC numpy array."""

    import imageio
    import numpy as np

    with local_path_for_uri(video_path) as local_path:
        frames = imageio.mimread(str(local_path), memtest=False)
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    arrays = []
    for frame in frames:
        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        arrays.append(array)
    return np.stack(arrays, axis=0)


def get_video_details(video_path: str | Path) -> tuple[int, float, float]:
    """Return ``(total_frames, fps, duration_seconds)`` for a local video."""

    from decord import VideoReader, cpu

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video path not found: {path}")
    if path.stat().st_size < 1024:
        raise ValueError(f"Video too short: {path}")
    reader = VideoReader(str(path), num_threads=-1, ctx=cpu(0))
    total_frames = len(reader)
    original_fps = float(reader.get_avg_fps())
    return total_frames, original_fps, total_frames / original_fps


def load_frames_from_video(
    video_path: str | Path,
    indices: Iterable[int],
    video_decode_backend: str = "decord",
    eval_: bool = True,
):
    """Load selected RGB frames into a torch tensor using decord or OpenCV."""

    import os

    import cv2
    import numpy as np
    import torch
    from decord import VideoReader, cpu

    path = str(video_path)
    frame_indices = [int(index) for index in indices]
    ext = os.path.splitext(path)[1].lower()
    if ext in {".gif", ".webm"} or video_decode_backend == "opencv":
        capture = cv2.VideoCapture(path)
        frames: dict[int, np.ndarray] = {}
        max_index = max(frame_indices)
        frame_id = 0
        ok = True
        while ok and frame_id <= max_index:
            ok, frame = capture.read()
            if ok and frame_id in frame_indices:
                frames[frame_id] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_id += 1
        capture.release()
        return torch.tensor(np.stack([frames[index] for index in frame_indices if index in frames]))

    reader = VideoReader(path) if eval_ else VideoReader(path, num_threads=1, ctx=cpu(0))
    batch = reader.get_batch(frame_indices)
    if isinstance(batch, torch.Tensor):
        return batch
    return torch.tensor(batch.asnumpy())


def read_video(video_path: str | Path, *, return_metadata: bool = True):
    """Decode a video into frames and optional metadata."""

    import imageio
    import numpy as np

    with local_path_for_uri(video_path) as local_path:
        reader = imageio.get_reader(str(local_path))
        frames = [np.asarray(frame) for frame in reader]
        metadata = reader.get_meta_data()
    if len(frames) == 0:
        raise ValueError(f"No frames found in video: {video_path}")
    stacked = np.stack(frames, axis=0)
    return (stacked, metadata) if return_metadata else stacked


def save_video_frames(video_frames, output_path: str | Path, fps: int = 16, **kwargs) -> None:
    """Write a THWC uint8 frame array/list to a video path or URI."""

    write_video(video_frames, output_path, fps=fps, **kwargs)


def save_video_h264(
    video_frames,
    output_path: str | Path,
    *,
    fps: float = 16.0,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """Write THWC RGB frames as a local H.264/yuv420p MP4 with system FFmpeg.

    This explicit path is useful for inference runtimes that require H.264 but
    should not depend on ImageIO's optional ``imageio-ffmpeg`` plugin.
    """

    import shutil
    import subprocess

    frames = coerce_video_frames(video_frames)
    if int(frames.shape[0]) == 0:
        return
    if parse_uri_scheme(output_path) != "file":
        raise ValueError("save_video_h264 currently supports local output paths only")

    target = uri_to_local_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise RuntimeError("ffmpeg not found in PATH; cannot encode H.264 output video")

    frame_count, height, width, _ = frames.shape
    command = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        assert process.stdin is not None
        for index in range(frame_count):
            process.stdin.write(frames[index].tobytes())
        process.stdin.close()
        assert process.stderr is not None
        stderr = process.stderr.read()
        process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg H.264 encode failed for {target}: {message}")


def save_image_or_video_tensor(
    tensor,
    save_path,
    *,
    fps: int = 24,
    quality: int | None = None,
    ffmpeg_params: list[str] | None = None,
    value_range: str | tuple[float, float] = "auto",
    image_format: str = "JPEG",
    video_format: str = "mp4",
    **kwargs,
) -> str | None:
    """Save a normalized ``[C,T,H,W]`` or ``[B,C,T,H,W]`` tensor as image/video.

    A single-frame tensor is saved as an image; multi-frame tensors are saved as
    videos. Local paths and URI-like targets supported by ``worldfoundry.core.io``
    storage helpers are both accepted.
    """

    from io import BytesIO

    from PIL import Image

    frames = video_tensor_to_uint8_frames(tensor, value_range=value_range)
    target = save_path
    is_file_obj = hasattr(target, "write")

    if frames.shape[0] == 1:
        image = Image.fromarray(frames[0][..., :3]).convert("RGB")
        if is_file_obj:
            image.save(target, format=image_format, quality=quality or 85, **kwargs)
            return None

        path_text = str(target)
        if not Path(path_text).suffix:
            path_text = f"{path_text}.jpg"
        if parse_uri_scheme(path_text) == "file":
            output_path = uri_to_local_path(path_text)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format=image_format, quality=quality or 85, **kwargs)
        else:
            buffer = BytesIO()
            image.save(buffer, format=image_format, quality=quality or 85, **kwargs)
            write_binary_uri(path_text, buffer)
        return path_text

    if is_file_obj:
        suffix = f".{video_format.lstrip('.')}"
        video_kwargs = dict(kwargs)
        if ffmpeg_params is not None:
            video_kwargs["ffmpeg_params"] = ffmpeg_params
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
            write_video(
                frames,
                handle.name,
                fps=fps,
                quality=quality,
                format=video_format,
                **video_kwargs,
            )
            handle.seek(0)
            target.write(handle.read())
        return None

    path_text = str(target)
    if not Path(path_text).suffix:
        path_text = f"{path_text}.mp4"
    video_kwargs = dict(kwargs)
    if ffmpeg_params is not None:
        video_kwargs["ffmpeg_params"] = ffmpeg_params
    write_video(
        frames,
        path_text,
        fps=fps,
        quality=quality,
        format=video_format,
        **video_kwargs,
    )
    return path_text


def write_video_torchvision(
    filename: str | Path,
    video_array: Any,
    fps: float,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Write RGB video frames with a ``torchvision.io.write_video``-compatible signature."""

    try:
        from torchvision.io import write_video as torchvision_write_video
    except (ImportError, AttributeError, RuntimeError):
        torchvision_write_video = None

    if torchvision_write_video is not None:
        torchvision_write_video(str(filename), video_array, fps, *args, **kwargs)
        return

    import cv2
    import numpy as np

    frames = video_array.detach().cpu().numpy() if hasattr(video_array, "detach") else np.asarray(video_array)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("write_video_torchvision expects frames shaped [T, H, W, C]")
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise OSError(f"failed to open video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_video(
    video_frames,
    output_path: str | Path,
    *,
    fps: int = 16,
    quality: int | None = None,
    format: str | None = None,
    **kwargs,
) -> None:
    """Write a THWC video array/list to a local path or remote URI."""

    import imageio

    frames = coerce_video_frames(video_frames)
    write_kwargs = {"fps": fps, "macro_block_size": 1, **kwargs}
    if quality is not None:
        write_kwargs["quality"] = quality

    if parse_uri_scheme(output_path) == "file":
        target = uri_to_local_path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(target), frames, format=format, **write_kwargs)
        return

    suffix = Path(str(output_path)).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        imageio.mimsave(handle.name, frames, format=format, **write_kwargs)
        handle.seek(0)
        write_binary_uri(output_path, handle.read())


def materialize_video_input(
    video_input,
    output_dir: Optional[str] = None,
    filename: str = "input.mp4",
    fps: int = 24,
) -> str:
    """Return a local video path, materializing in-memory inputs when needed."""

    import os

    if isinstance(video_input, (str, os.PathLike)):
        candidate = uri_to_local_path(video_input) if parse_uri_scheme(video_input) == "file" else None
        if (
            candidate is not None
            and candidate.exists()
            and candidate.is_file()
            and candidate.suffix.lower() in VIDEO_EXTENSIONS
        ):
            return str(candidate.resolve())

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="worldfoundry_video_")
    output_path = Path(output_dir).expanduser().resolve() / filename
    frames = coerce_video_frames(video_input)
    save_video_frames(frames, str(output_path), fps=fps)
    return str(output_path)


def save_videos_grid(videos, path: str, rescale=False, n_rows=6, fps=8):
    """Save a batch of BCTHW video tensors as a single grid video."""

    import os

    import imageio
    import numpy as np
    import torchvision
    from einops import rearrange

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


__all__ = [
    "VIDEO_EXTENSIONS",
    "coerce_video_frames",
    "extract_frames_from_video_url",
    "get_video_details",
    "load_frames_from_video",
    "load_video_frames",
    "materialize_video_input",
    "read_image_as_video_tensor",
    "read_video",
    "resize_video_tensor_to_resolution",
    "save_image_or_video_tensor",
    "save_video_frames",
    "save_videos_grid",
    "video_tensor_to_uint8_frames",
    "write_video",
    "write_video_torchvision",
]
