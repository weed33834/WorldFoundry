"""Vid2World runtime video helpers."""

from einops import rearrange
from torch.nn import functional as F


def none_or_int(value: str):
    if value.lower() == "none":
        return None
    return int(value)


def resize_video(video_tensor, new_size):
    """Resize a tensor of shape (b, t, h, w, c) to (b, t, new_h, new_w, c)."""
    original_dtype = video_tensor.dtype
    video_tensor = video_tensor.float()
    batch_size, num_frames, _, _, _ = video_tensor.shape
    video_tensor = rearrange(video_tensor, "b t h w c -> (b t) c h w")
    resized_tensor = F.interpolate(video_tensor, size=new_size, mode="bilinear", align_corners=False)
    resized_tensor = rearrange(resized_tensor, "(b t) c h w -> b t h w c", b=batch_size, t=num_frames)
    return resized_tensor.to(original_dtype)
