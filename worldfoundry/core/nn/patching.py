"""Generic 2D patchify/unpatchify helpers for tensor-like arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

ImageLayout = Literal["nchw", "nhwc"]


@dataclass(frozen=True)
class PatchGridSpec:
    """Shape contract needed to invert a 2D image patchification."""

    original_shape: tuple[int, ...]
    patch_size: tuple[int, int]
    layout: ImageLayout = "nchw"

    def __post_init__(self) -> None:
        shape = tuple(int(item) for item in self.original_shape)
        patch = _patch_size_2d(self.patch_size)
        layout = _normalize_layout(self.layout)
        if len(shape) < 3:
            raise ValueError("PatchGridSpec requires at least channel and 2D spatial dimensions.")
        h, w = _spatial_shape(shape, layout)
        if h % patch[0] or w % patch[1]:
            raise ValueError(f"spatial shape {(h, w)!r} is not divisible by patch_size {patch!r}.")
        object.__setattr__(self, "original_shape", shape)
        object.__setattr__(self, "patch_size", patch)
        object.__setattr__(self, "layout", layout)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.original_shape[:-3]

    @property
    def channels(self) -> int:
        return self.original_shape[-3] if self.layout == "nchw" else self.original_shape[-1]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return _spatial_shape(self.original_shape, self.layout)

    @property
    def grid_shape(self) -> tuple[int, int]:
        h, w = self.spatial_shape
        ph, pw = self.patch_size
        return h // ph, w // pw

    @property
    def patch_vector_size(self) -> int:
        ph, pw = self.patch_size
        return self.channels * ph * pw

    @property
    def patch_count(self) -> int:
        gh, gw = self.grid_shape
        return gh * gw


def patchify_image(value: Any, patch_size: int | Sequence[int], *, layout: ImageLayout = "nchw") -> tuple[Any, PatchGridSpec]:
    """Convert an image tensor into flattened 2D patch tokens.

    ``layout="nchw"`` expects ``(..., C, H, W)`` and ``layout="nhwc"`` expects
    ``(..., H, W, C)``. The returned token tensor has shape
    ``(..., grid_h * grid_w, C * patch_h * patch_w)``.
    """

    shape = _shape_tuple(value)
    spec = PatchGridSpec(original_shape=shape, patch_size=_patch_size_2d(patch_size), layout=layout)
    batch = spec.batch_shape
    ph, pw = spec.patch_size
    gh, gw = spec.grid_shape
    c = spec.channels
    b = len(batch)

    if spec.layout == "nchw":
        reshaped = value.reshape(*batch, c, gh, ph, gw, pw)
        transposed = _transpose(reshaped, (*range(b), b + 1, b + 3, b, b + 2, b + 4))
    else:
        reshaped = value.reshape(*batch, gh, ph, gw, pw, c)
        transposed = _transpose(reshaped, (*range(b), b, b + 2, b + 1, b + 3, b + 4))
    return transposed.reshape(*batch, gh * gw, c * ph * pw), spec


def unpatchify_image(patches: Any, spec: PatchGridSpec) -> Any:
    """Invert ``patchify_image`` using the returned ``PatchGridSpec``."""

    patch_shape = _shape_tuple(patches)
    expected_shape = (*spec.batch_shape, spec.patch_count, spec.patch_vector_size)
    if patch_shape != expected_shape:
        raise ValueError(f"patch tensor shape {patch_shape!r} does not match expected shape {expected_shape!r}.")

    batch = spec.batch_shape
    ph, pw = spec.patch_size
    gh, gw = spec.grid_shape
    c = spec.channels
    b = len(batch)

    if spec.layout == "nchw":
        reshaped = patches.reshape(*batch, gh, gw, c, ph, pw)
        transposed = _transpose(reshaped, (*range(b), b + 2, b, b + 3, b + 1, b + 4))
    else:
        reshaped = patches.reshape(*batch, gh, gw, ph, pw, c)
        transposed = _transpose(reshaped, (*range(b), b, b + 2, b + 1, b + 3, b + 4))
    return transposed.reshape(*spec.original_shape)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError("value must expose a shape attribute and reshape method.")
    if not callable(getattr(value, "reshape", None)):
        raise TypeError("value must expose a reshape method.")
    return tuple(int(item) for item in shape)


def _patch_size_2d(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        patch = (value, value)
    else:
        items = tuple(int(item) for item in value)
        if len(items) != 2:
            raise ValueError("patch_size must be an int or a two-item sequence.")
        patch = items
    if patch[0] <= 0 or patch[1] <= 0:
        raise ValueError("patch_size values must be positive.")
    return patch


def _normalize_layout(value: str) -> ImageLayout:
    normalized = str(value).strip().lower()
    if normalized not in {"nchw", "nhwc"}:
        raise ValueError("layout must be 'nchw' or 'nhwc'.")
    return normalized  # type: ignore[return-value]


def _spatial_shape(shape: tuple[int, ...], layout: ImageLayout) -> tuple[int, int]:
    if layout == "nchw":
        return shape[-2], shape[-1]
    return shape[-3], shape[-2]


def _transpose(value: Any, axes: tuple[int, ...]) -> Any:
    permute = getattr(value, "permute", None)
    if callable(permute):
        return permute(*axes)
    transpose = getattr(value, "transpose", None)
    if callable(transpose):
        return transpose(axes)
    raise TypeError("value must expose a transpose or permute method.")


__all__ = ["PatchGridSpec", "patchify_image", "unpatchify_image"]
