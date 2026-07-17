"""Low-latency WebRTC runtime for the standalone interactive world frontend.

The offline Studio run contract is intentionally not used after session setup.
One model context stays resident on one execution thread, sparse input edges are
resampled at chunk boundaries, and decoded RGB frames move through a bounded
in-memory queue into one long-lived WebRTC video track.
"""

from __future__ import annotations

import asyncio
import functools
import io
import json
import os
import re
import time
import traceback
import uuid
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote

import numpy as np
from PIL import Image

from worldfoundry.core.realtime import RealtimeSpec
from worldfoundry.studio.catalog import CatalogEntry
from worldfoundry.studio.execution import (
    PreparedInputs,
    StudioManager,
    _normalize_frame_list,
    _to_uint8_rgb,
)
from worldfoundry.studio.launch_config import StudioLaunchConfig

SUPPORTED_CONTROL_KEYS = frozenset({"w", "a", "s", "d", "i", "j", "k", "l"})
_DREAMX_WORLD_MODEL_ID = "dreamx-world-5b-cam"
_PROMPT_BOUNDARY_MODEL_IDS = frozenset(
    {
        "dreamx-world-5b-cam",
        "helios",
        "lingbot-world",
        "lingbot-world-v2",
        "matrix-game-3",
        "sana-wm",
    }
)
KEY_ALIASES = {
    "arrowup": "i",
    "arrowdown": "k",
    "arrowleft": "j",
    "arrowright": "l",
}

_TEXT_EVENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_MAX_TEXT_EVENTS = 12
_MAX_TEXT_EVENT_PROMPT = 1000
_MIN_OUTPUT_WIDTH = 160
_MIN_OUTPUT_HEIGHT = 90
_MAX_OUTPUT_WIDTH = 1920
_MAX_OUTPUT_HEIGHT = 1920
_MAX_OUTPUT_PIXELS = 1920 * 1080


def normalize_text_events(value: Any) -> list[dict[str, str]]:
    """Validate the small, user-authored event catalog carried by a session."""

    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("text_events must be a JSON list.")
    if len(value) > _MAX_TEXT_EVENTS:
        raise ValueError(f"text_events supports at most {_MAX_TEXT_EVENTS} events.")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"text_events[{index - 1}] must be an object.")
        event_id = str(item.get("event_id") or item.get("id") or "").strip()
        if not _TEXT_EVENT_ID.fullmatch(event_id):
            raise ValueError(
                f"text_events[{index - 1}].event_id must use 1-64 letters, numbers, or _.:-."
            )
        if event_id in seen:
            raise ValueError(f"Duplicate text event id: {event_id}.")
        label = str(item.get("label") or event_id).strip()
        prompt = str(item.get("prompt") or "").strip()
        category = str(item.get("category") or "event").strip()
        if not label or len(label) > 64:
            raise ValueError(f"Text event {event_id!r} label must contain 1-64 characters.")
        if not prompt or len(prompt) > _MAX_TEXT_EVENT_PROMPT:
            raise ValueError(
                f"Text event {event_id!r} prompt must contain 1-{_MAX_TEXT_EVENT_PROMPT} characters."
            )
        if not category or len(category) > 64:
            raise ValueError(f"Text event {event_id!r} category must contain 1-64 characters.")
        seen.add(event_id)
        result.append(
            {
                "event_id": event_id,
                "label": label,
                "prompt": prompt,
                "category": category,
            }
        )
    return result


def normalize_output_resolution(value: Any) -> tuple[int, int] | None:
    """Return an even transport resolution, or ``None`` for model-native output."""

    if value in (None, "", "native"):
        return None
    if isinstance(value, str):
        match = re.fullmatch(r"\s*(\d+)\s*[xX]\s*(\d+)\s*", value)
        if not match:
            raise ValueError("output_resolution must be 'native' or WIDTHxHEIGHT.")
        width, height = (int(match.group(1)), int(match.group(2)))
    elif isinstance(value, Mapping):
        mode = str(value.get("mode") or "").strip().lower()
        if mode == "native":
            return None
        try:
            width = int(value.get("width") or 0)
            height = int(value.get("height") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("output_resolution width and height must be integers.") from exc
    else:
        raise ValueError("output_resolution must be an object, WIDTHxHEIGHT, or 'native'.")
    if not (_MIN_OUTPUT_WIDTH <= width <= _MAX_OUTPUT_WIDTH):
        raise ValueError(
            f"output width must be between {_MIN_OUTPUT_WIDTH} and {_MAX_OUTPUT_WIDTH}."
        )
    if not (_MIN_OUTPUT_HEIGHT <= height <= _MAX_OUTPUT_HEIGHT):
        raise ValueError(
            f"output height must be between {_MIN_OUTPUT_HEIGHT} and {_MAX_OUTPUT_HEIGHT}."
        )
    if width % 2 or height % 2:
        raise ValueError("output width and height must be even for realtime video encoding.")
    return width, height


@dataclass(frozen=True, slots=True)
class OutputResolutionSnapshot:
    """Immutable resolution decision for one generated transport chunk."""

    dimensions: tuple[int, int] | None
    source_dimensions: tuple[int, int] | None
    revision: int

    def to_payload(self) -> dict[str, Any]:
        if self.dimensions is not None:
            return {
                "mode": "fixed",
                "width": self.dimensions[0],
                "height": self.dimensions[1],
            }
        payload: dict[str, Any] = {"mode": "native"}
        if self.source_dimensions is not None:
            payload.update(
                width=self.source_dimensions[0],
                height=self.source_dimensions[1],
            )
        return payload


@dataclass(slots=True)
class OutputResolutionState:
    """Per-session output boundary with source-aware anti-upscale checks."""

    width: int | None = None
    height: int | None = None
    max_width: int | None = None
    max_height: int | None = None
    max_pixels: int = _MAX_OUTPUT_PIXELS
    source_width: int | None = None
    source_height: int | None = None
    revision: int = 0

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        maximum: tuple[int, int] | None = None,
        max_pixels: int | None = None,
    ) -> "OutputResolutionState":
        state = cls(
            max_width=maximum[0] if maximum is not None else None,
            max_height=maximum[1] if maximum is not None else None,
            max_pixels=min(max(int(max_pixels or _MAX_OUTPUT_PIXELS), 1), _MAX_OUTPUT_PIXELS),
        )
        state._set(value, increment_revision=False)
        return state

    @property
    def dimensions(self) -> tuple[int, int] | None:
        if self.width is None or self.height is None:
            return None
        return self.width, self.height

    @property
    def source_dimensions(self) -> tuple[int, int] | None:
        if self.source_width is None or self.source_height is None:
            return None
        return self.source_width, self.source_height

    def _validate_dimensions(self, dimensions: tuple[int, int] | None) -> None:
        if dimensions is None:
            return
        width, height = dimensions
        if self.max_width is not None and width > self.max_width:
            raise ValueError(
                f"output width {width} exceeds the model-native width {self.max_width}; "
                "realtime output does not upscale generated frames."
            )
        if self.max_height is not None and height > self.max_height:
            raise ValueError(
                f"output height {height} exceeds the model-native height {self.max_height}; "
                "realtime output does not upscale generated frames."
            )
        if width * height > self.max_pixels:
            raise ValueError(
                f"output resolution {width}x{height} exceeds the realtime pixel budget "
                f"of {self.max_pixels} pixels."
            )
        source = self.source_dimensions
        if source is not None and (width > source[0] or height > source[1]):
            raise ValueError(
                f"output resolution {width}x{height} exceeds the generated frame "
                f"{source[0]}x{source[1]}; realtime output does not upscale."
            )

    def _set(self, value: Any, *, increment_revision: bool) -> bool:
        dimensions = normalize_output_resolution(value)
        self._validate_dimensions(dimensions)
        if dimensions == self.dimensions:
            return False
        self.width, self.height = dimensions or (None, None)
        if increment_revision:
            self.revision += 1
        return True

    def update(self, value: Any) -> bool:
        return self._set(value, increment_revision=True)

    def observe_source(self, source: np.ndarray | None) -> bool:
        """Bind the real model output and safely downgrade an invalid early request."""

        if source is None or source.ndim < 2:
            return False
        source_dimensions = (int(source.shape[1]), int(source.shape[0]))
        self.source_width, self.source_height = source_dimensions
        try:
            self._validate_dimensions(self.dimensions)
        except ValueError:
            self.width = None
            self.height = None
            self.revision += 1
            return True
        return False

    def snapshot(self) -> OutputResolutionSnapshot:
        return OutputResolutionSnapshot(
            dimensions=self.dimensions,
            source_dimensions=self.source_dimensions,
            revision=self.revision,
        )

    def to_payload(self) -> dict[str, Any]:
        return self.snapshot().to_payload()


def _entry_output_resolution(entry: CatalogEntry) -> tuple[int, int] | None:
    values = entry.default_call_kwargs
    try:
        width = int(values.get("width") or values.get("user_width") or 0)
        height = int(values.get("height") or values.get("user_height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return width, height
    resolution = values.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        try:
            # Catalog resolution tuples follow the model convention (height, width).
            height, width = (int(resolution[0]), int(resolution[1]))
        except (TypeError, ValueError):
            return None
        if width > 0 and height > 0:
            return width, height
    size = values.get("size")
    if isinstance(size, str):
        # World-model integrations commonly spell size as HEIGHT*WIDTH.
        match = re.fullmatch(r"\s*(\d+)\s*\*\s*(\d+)\s*", size)
        if match:
            height, width = (int(match.group(1)), int(match.group(2)))
            if width > 0 and height > 0:
                return width, height
    return None


def _entry_output_pixel_budget(entry: CatalogEntry) -> int:
    native = _entry_output_resolution(entry)
    if native is not None:
        return min(native[0] * native[1], _MAX_OUTPUT_PIXELS)
    try:
        max_area = int(entry.default_call_kwargs.get("max_area") or 0)
    except (TypeError, ValueError):
        max_area = 0
    return min(max_area, _MAX_OUTPUT_PIXELS) if max_area > 0 else _MAX_OUTPUT_PIXELS


def _new_output_resolution_state(entry: CatalogEntry, value: Any) -> OutputResolutionState:
    return OutputResolutionState.from_value(
        value,
        maximum=_entry_output_resolution(entry),
        max_pixels=_entry_output_pixel_budget(entry),
    )


def _output_resolution_options(entry: CatalogEntry) -> list[dict[str, Any]]:
    native = _entry_output_resolution(entry)
    options: list[dict[str, Any]] = [
        {
            "mode": "native",
            "label": (
                f"Native · {native[0]}×{native[1]}" if native is not None else "Native"
            ),
            **(
                {"width": native[0], "height": native[1]}
                if native is not None
                else {}
            ),
        }
    ]
    candidates: list[tuple[int, int]] = []
    if native is not None:
        for scale in (0.75, 0.5):
            width = max(int(round(native[0] * scale / 2.0)) * 2, _MIN_OUTPUT_WIDTH)
            height = max(int(round(native[1] * scale / 2.0)) * 2, _MIN_OUTPUT_HEIGHT)
            candidates.append((width, height))
    # Without a model-owned native size, guessed presets can accidentally
    # upscale the first generated frame. Keep only native until the runtime
    # has observed a real source size.
    pixel_budget = _entry_output_pixel_budget(entry)
    for width, height in dict.fromkeys(candidates):
        if not (
            _MIN_OUTPUT_WIDTH <= width <= _MAX_OUTPUT_WIDTH
            and _MIN_OUTPUT_HEIGHT <= height <= _MAX_OUTPUT_HEIGHT
        ):
            continue
        if width * height > pixel_budget:
            continue
        if native is not None and (width > native[0] or height > native[1]):
            continue
        options.append(
            {
                "mode": "fixed",
                "width": width,
                "height": height,
                "label": f"Stream · {width}×{height}",
            }
        )
    return options


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default)) or default), minimum)
    except Exception:
        return max(default, minimum)


def _realtime_frame_budget(entry: CatalogEntry, requested: int) -> int:
    """Resolve the bootstrap budget before a model reports its native spec."""

    if "queued-segment-generation" in entry.tags:
        # Queued diffusion segments are not latency-clamped realtime chunks.
        # Their native frame count is part of the model's quality/continuation
        # contract and must be known before the first result reports its spec.
        native_frames = entry.default_call_kwargs.get("num_frames")
        if native_frames is not None:
            try:
                return max(int(native_frames), 1)
            except (TypeError, ValueError):
                pass
    return max(int(requested), 1)


def _default_realtime_chunk_frames(entry: CatalogEntry) -> int:
    # DreamX uses 1 + 4k model windows. Five model frames are the smallest
    # native control interval and avoid silently inheriting the generic
    # nine-frame latency profile.
    return 5 if entry.model_id == _DREAMX_WORLD_MODEL_ID else 9


def _default_realtime_inference_steps(
    entry: CatalogEntry,
    launch_config: StudioLaunchConfig,
) -> int | None:
    configured_steps = os.getenv("WORLDFOUNDRY_REALTIME_INFERENCE_STEPS", "").strip()
    if configured_steps:
        return _env_int("WORLDFOUNDRY_REALTIME_INFERENCE_STEPS", 4)
    if entry.model_id == _DREAMX_WORLD_MODEL_ID:
        return 4
    if entry.model_id == "lingbot-world" and launch_config.variant_id == "fast":
        return 4
    return None


def _ice_server_payload() -> list[dict[str, Any]]:
    """Parse browser/aiortc-compatible STUN/TURN configuration."""

    raw = os.getenv("WORLDFOUNDRY_REALTIME_ICE_SERVERS_JSON", "").strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("WORLDFOUNDRY_REALTIME_ICE_SERVERS_JSON is not valid JSON.") from exc
    if isinstance(decoded, Mapping):
        decoded = decoded.get("iceServers")
    if not isinstance(decoded, list):
        raise RuntimeError("Realtime ICE config must be a JSON list or an {iceServers: [...]} object.")
    result: list[dict[str, Any]] = []
    for item in decoded:
        if not isinstance(item, Mapping) or not item.get("urls"):
            raise RuntimeError("Each realtime ICE server must define a non-empty 'urls' value.")
        server = {"urls": item["urls"]}
        for key in ("username", "credential"):
            if item.get(key) is not None:
                server[key] = str(item[key])
        result.append(server)
    return result


def normalize_control_key(key: str) -> str:
    normalized = str(key or "").strip().lower()
    return KEY_ALIASES.get(normalized, normalized)


@dataclass(slots=True)
class RealtimeControlState:
    """Pressed-key state with last-pressed-wins conflict resolution."""

    pressed: set[str] = field(default_factory=set)
    _order: dict[str, int] = field(default_factory=dict)
    _sequence: int = 0

    def apply(self, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in SUPPORTED_CONTROL_KEYS:
            return False
        event = str(event or "").strip().lower()
        if event == "keydown":
            self.pressed.add(normalized)
            self._sequence += 1
            self._order[normalized] = self._sequence
            return True
        if event == "keyup":
            self.pressed.discard(normalized)
            self._order.pop(normalized, None)
            return True
        return False

    def _latest(self, keys: tuple[str, ...]) -> str | None:
        active = [key for key in keys if key in self.pressed]
        return max(active, key=lambda key: self._order.get(key, -1), default=None)

    def effective(self) -> frozenset[str]:
        return frozenset(
            key
            for key in (
                self._latest(("w", "s")),
                self._latest(("a", "d")),
                self._latest(("i", "k")),
                self._latest(("j", "l")),
            )
            if key is not None
        )


ControlSegment = tuple[float, float, frozenset[str]]


class RealtimeControlResampler:
    """Resample timestamped input edges into a model chunk timeline."""

    def __init__(self, *, fps: int, start_time: float = 0.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self.fps = int(fps)
        self.dt = 1.0 / float(fps)
        self.next_chunk_start = float(start_time)
        self._events: deque[tuple[float, str, str]] = deque()
        self._state = RealtimeControlState()

    def on_edge(self, *, arrival_time: float, event: str, key: str) -> bool:
        normalized = normalize_control_key(key)
        if normalized not in SUPPORTED_CONTROL_KEYS or event not in {"keydown", "keyup"}:
            return False
        self._events.append((float(arrival_time), event, normalized))
        return True

    def sample_chunk(self, num_frames: int, *, wall_time: float) -> list[ControlSegment]:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        duration = num_frames * self.dt
        if self.next_chunk_start <= 0.0 or wall_time - self.next_chunk_start > duration:
            self.next_chunk_start = float(wall_time)
        start = self.next_chunk_start
        end = start + duration

        while self._events and self._events[0][0] < start:
            _, event, key = self._events.popleft()
            self._state.apply(event, key)

        segments: list[ControlSegment] = []
        cursor = start
        effective = self._state.effective()
        while self._events and self._events[0][0] <= end:
            event_time, event, key = self._events.popleft()
            if event_time > cursor:
                segments.append((cursor, event_time, effective))
            self._state.apply(event, key)
            effective = self._state.effective()
            cursor = max(cursor, event_time)
        if cursor < end or not segments:
            segments.append((cursor, end, effective))
        self.next_chunk_start = end
        return segments

    def reset(self, *, start_time: float) -> None:
        self._events.clear()
        self._state = RealtimeControlState()
        self.next_chunk_start = float(start_time)

    @property
    def effective_keys(self) -> frozenset[str]:
        # Apply edges that have already arrived so key release can stop the
        # producer immediately after the in-flight chunk finishes.
        now = time.monotonic()
        while self._events and self._events[0][0] <= now:
            _, event, key = self._events.popleft()
            self._state.apply(event, key)
        return self._state.effective()


def interactions_from_keys(keys: frozenset[str]) -> list[str]:
    tokens: list[str] = []
    for key, token in (
        ("w", "forward"),
        ("s", "backward"),
        ("a", "left"),
        ("d", "right"),
        ("i", "camera_up"),
        ("k", "camera_down"),
        ("j", "camera_l"),
        ("l", "camera_r"),
    ):
        if key in keys:
            tokens.append(token)
    return tokens


def interactions_from_segments(segments: list[ControlSegment]) -> list[str]:
    """Return the most recent non-idle control state in a sampled chunk."""

    for _, _, keys in reversed(segments):
        interactions = interactions_from_keys(keys)
        if interactions:
            return interactions
    return []


class LatestFrameBuffer:
    """Bounded frame queue with short backpressure, then stale-frame eviction."""

    def __init__(self, *, maxsize: int, backpressure_ms: int = 0) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = int(maxsize)
        self.backpressure_s = max(float(backpressure_ms), 0.0) / 1000.0
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=maxsize)
        self.dropped_frames = 0
        self.last_enqueue_ms = 0.0
        self.closed = False

    def qsize(self) -> int:
        return self._queue.qsize()

    async def put_chunk(self, frames: list[np.ndarray]) -> int:
        if self.closed:
            return 0
        loop = asyncio.get_running_loop()
        started = loop.time()
        deadline = started + self.backpressure_s
        accepted = 0
        for frame in frames:
            # Result normalization already moved tensors to CPU and converted
            # them to RGB uint8 on the runtime worker. Keep the event loop's
            # media handoff to a cheap contiguous-array check.
            rgb = np.ascontiguousarray(frame)
            if self._queue.full() and self.backpressure_s > 0:
                remaining = deadline - loop.time()
                if remaining > 0:
                    try:
                        await asyncio.wait_for(self._queue.put(rgb), timeout=remaining)
                        accepted += 1
                        continue
                    except TimeoutError:
                        pass
            while self._queue.full():
                try:
                    self._queue.get_nowait()
                    self.dropped_frames += 1
                except asyncio.QueueEmpty:
                    break
            self._queue.put_nowait(rgb)
            accepted += 1
        self.last_enqueue_ms = (loop.time() - started) * 1000.0
        return accepted

    async def get(self) -> np.ndarray:
        frame = await self._queue.get()
        if frame is None:
            raise EOFError("frame buffer closed")
        return frame

    def get_nowait(self) -> np.ndarray:
        frame = self._queue.get_nowait()
        if frame is None:
            raise EOFError("frame buffer closed")
        return frame

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(None)


def realtime_frames_from_result(result: Any) -> list[np.ndarray]:
    """Extract in-memory RGB frames without invoking artifact exporters."""

    if isinstance(result, Iterator):
        close = getattr(result, "close", None)
        try:
            # Materialize at the actual frame consumer so the runtime can keep
            # generator-based integrations lazy and pinned up to this boundary.
            result = list(result)
        finally:
            if callable(close):
                close()

    candidates: list[Any] = []
    if isinstance(result, Mapping):
        for key in ("sr_videos", "videos", "frames", "video", "output", "images"):
            if key in result:
                candidates.append(result[key])
    else:
        candidates.append(result)

    for candidate in candidates:
        if isinstance(candidate, Image.Image):
            return [_to_uint8_rgb(candidate)]
        frames = _normalize_frame_list(candidate)
        if frames:
            return [np.ascontiguousarray(frame) for frame in frames]
    raise RuntimeError(
        "Realtime stream returned no in-memory RGB frames. The model integration "
        "must return a tensor/array/image chunk instead of only an artifact path."
    )


def _resize_rgb_frame(
    frame: np.ndarray,
    output_resolution: tuple[int, int] | None,
) -> np.ndarray:
    """Prepare one transport frame without ever enlarging model output."""

    rgb = np.ascontiguousarray(frame)
    if output_resolution is None or (rgb.shape[1], rgb.shape[0]) == output_resolution:
        return rgb
    width, height = output_resolution
    source_height, source_width = rgb.shape[:2]
    if (
        width > _MAX_OUTPUT_WIDTH
        or height > _MAX_OUTPUT_HEIGHT
        or width * height > _MAX_OUTPUT_PIXELS
    ):
        raise ValueError(
            f"realtime transport resolution {width}x{height} exceeds its safe output budget."
        )
    if width > source_width or height > source_height:
        raise ValueError(
            f"realtime transport cannot upscale {source_width}x{source_height} "
            f"model output to {width}x{height}."
        )
    scale = min(width / source_width, height / source_height)
    fitted = (
        max(min(int(round(source_width * scale)), width), 1),
        max(min(int(round(source_height * scale)), height), 1),
    )
    resized = Image.fromarray(rgb, mode="RGB").resize(
        fitted,
        resample=Image.Resampling.BILINEAR,
    )
    if fitted == output_resolution:
        return np.ascontiguousarray(np.asarray(resized, dtype=np.uint8))
    canvas = Image.new("RGB", output_resolution, "black")
    canvas.paste(resized, ((width - fitted[0]) // 2, (height - fitted[1]) // 2))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8))


def _resize_rgb_frames(
    frames: list[np.ndarray],
    *,
    output_resolution: tuple[int, int] | None,
) -> list[np.ndarray]:
    return [_resize_rgb_frame(frame, output_resolution) for frame in frames]


def _encode_jpeg_frames(
    frames: list[np.ndarray],
    *,
    quality: int,
    subsampling: int = 1,
    output_resolution: tuple[int, int] | None = None,
) -> list[bytes]:
    """Encode a generated chunk for the same-port WebSocket fallback."""

    packets: list[bytes] = []
    for frame in frames:
        output = io.BytesIO()
        image = Image.fromarray(
            _resize_rgb_frame(frame, output_resolution),
            mode="RGB",
        )
        image.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=False,
            subsampling=subsampling,
        )
        packets.append(output.getvalue())
    return packets


def _validate_control_video_frames(path: str, *, expected_frames: int) -> int:
    """Decode enough control frames to reject a bad EXTEND transaction early."""

    from av import open as av_open

    source = Path(path).expanduser()
    if not source.is_file():
        raise ValueError(f"Control video does not exist: {source}")
    try:
        with av_open(str(source)) as container:
            if not container.streams.video:
                raise ValueError(f"Control input has no video stream: {source.name}")
            count = 0
            for _frame in container.decode(video=0):
                count += 1
                if count >= expected_frames:
                    return count
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot decode control video {source.name}: {exc}") from exc
    raise ValueError(
        f"Control video {source.name} has {count} decoded frame(s); "
        f"this segment needs at least {expected_frames}."
    )


def _realtime_overrides(
    entry: CatalogEntry,
    base: Mapping[str, Any],
    *,
    inference_steps: int | None = None,
) -> dict[str, Any]:
    """Clamp catalog demo settings to a latency-oriented chunk profile."""

    values = dict(base)
    frame_budget = _realtime_frame_budget(
        entry,
        _env_int(
            "WORLDFOUNDRY_REALTIME_CHUNK_FRAMES",
            _default_realtime_chunk_frames(entry),
        ),
    )
    params = set(entry.call_params) | set(entry.stream_params) | set(values)

    if "num_frames" in params:
        values["num_frames"] = frame_budget
    if "video_length" in params:
        values["video_length"] = frame_budget
    if "frame_num" in params:
        values["frame_num"] = frame_budget
    if "num_chunks" in params:
        values["num_chunks"] = 1
    if "num_iterations" in params:
        values["num_iterations"] = 1
    if inference_steps is not None:
        for key in ("num_inference_steps", "inference_steps", "sampling_steps", "infer_steps"):
            if key in params:
                values[key] = inference_steps
    for key in ("visualize_ops", "visualize_warning", "show_progress"):
        if key in params:
            values[key] = False
    # A live session must always consume the DataChannel control state.  Several
    # catalog presets intentionally point at offline benchmark action files; if
    # those values leak into the resident runtime the UI looks responsive while
    # the model is actually replaying a fixed trajectory.
    if "action_path" in params:
        values["action_path"] = None
    if "official_bench_actions" in params:
        values["official_bench_actions"] = False
    values.pop("return_dict", None)

    return values


def _interactive_call_kwargs(
    base: Mapping[str, Any],
    interactions: list[str],
    *,
    seed: int,
    control_segments: list[ControlSegment] | None = None,
) -> dict[str, Any]:
    values = {**base, "seed": int(seed)}
    if control_segments is not None:
        values["realtime_segments"] = [
            {
                "duration": max(float(end) - float(start), 0.0),
                "keys": sorted(keys),
            }
            for start, end, keys in control_segments
        ]
    if "interaction_speed" in values:
        configured_speed = values["interaction_speed"]
        if isinstance(configured_speed, (list, tuple)) and configured_speed:
            speed = configured_speed[0]
        elif configured_speed is None:
            speed = 0.2
        else:
            speed = configured_speed
        values["interaction_speed"] = [speed] * len(interactions)
    return values


class ResidentWorldRuntime:
    """One resident model context executed on one stable worker thread."""

    def __init__(
        self,
        *,
        manager: StudioManager,
        entry: CatalogEntry,
        launch_config: StudioLaunchConfig,
        fps: int,
        warmup_image_path: str = "",
        warmup_chunks: int = 0,
    ) -> None:
        self.manager = manager
        self.entry = entry
        self.launch_config = launch_config
        self.fps = fps
        self.warmup_image_path = warmup_image_path
        self.warmup_chunks = max(int(warmup_chunks), 0)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="world-realtime-runtime")
        self._base_request: PreparedInputs | None = None
        self._preload_future: asyncio.Future[Any] | None = None
        self._preload_error: str | None = None
        self.warmup_ms = 0.0
        self._configured = False
        self._first_stream_step = True
        self._seed_image: Image.Image | None = None
        self.last_generation_metrics: dict[str, Any] = {}
        self.queued_segment_generation = "queued-segment-generation" in entry.tags
        bootstrap_frames = _realtime_frame_budget(
            entry,
            _env_int(
                "WORLDFOUNDRY_REALTIME_CHUNK_FRAMES",
                _default_realtime_chunk_frames(entry),
            ),
        )
        if self.queued_segment_generation:
            self.realtime_spec = RealtimeSpec(
                fps=int(entry.default_call_kwargs.get("fps") or self.fps),
                first_chunk_frames=bootstrap_frames,
                steady_chunk_frames=bootstrap_frames,
                controls=("dense_depth_video", "sparse_pointmap_or_track_video"),
                transport="queued-segment-rgb",
                stateful=True,
            )
        else:
            self.realtime_spec = RealtimeSpec(
                fps=self.fps,
                first_chunk_frames=bootstrap_frames,
                steady_chunk_frames=bootstrap_frames,
            )

    def _accept_realtime_spec(self, result: Any) -> None:
        self.realtime_spec = RealtimeSpec.from_payload(
            result,
            fallback=self.realtime_spec,
        )
        # Playback, input resampling, and the model request must share the
        # cadence advertised by the resident adapter. Keeping the bootstrap
        # frontend FPS here made 12/17/24-FPS models play at a hard-coded 16.
        self.fps = self.realtime_spec.fps

    async def _run(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self._executor, call)

    def _build_request(
        self,
        *,
        prompt: str,
        image: Image.Image | None,
        input_path: str,
        video_path: str,
        dense_video_path: str = "",
        sparse_video_path: str = "",
    ) -> PreparedInputs:
        from worldfoundry.studio.visualization.backends.world import (
            _api_key,
            _launch_runtime_overrides,
        )

        load_text, call_text, model_ref = _launch_runtime_overrides(
            self.entry,
            self.launch_config,
            interactions_text="",
        )
        call_values = json.loads(call_text or "{}")
        if self.queued_segment_generation:
            # Do not apply the low-latency frame/step clamps to full-quality
            # queued segments. Launch call kwargs remain the explicit tuning
            # surface for users who intentionally choose another profile.
            if dense_video_path:
                call_values["dense_video"] = dense_video_path
            else:
                call_values.pop("dense_video", None)
            if sparse_video_path:
                call_values["sparse_video"] = sparse_video_path
            else:
                call_values.pop("sparse_video", None)
            call_values["return_dict"] = True
        else:
            call_values = _realtime_overrides(
                self.entry,
                call_values,
                inference_steps=_default_realtime_inference_steps(
                    self.entry,
                    self.launch_config,
                ),
            )
        return self.manager.prepare_inputs(
            entry=self.entry,
            prompt=prompt,
            input_path=input_path,
            image=image,
            video=video_path or None,
            last_frame=None,
            reference_files=None,
            interactions_text="",
            camera_view_text="",
            task_type=self.entry.default_task_type or "",
            intrinsics_text="",
            meta_path="",
            panorama_path="",
            scene_name="",
            fps=self.fps,
            num_frames=int(call_values.get("num_frames") or 0),
            call_kwargs_text=json.dumps(call_values),
            load_kwargs_text=load_text,
            model_ref=model_ref,
            backend=self.launch_config.backend or self.entry.default_backend or "auto",
            endpoint=self.launch_config.endpoint or self.entry.default_endpoint or "",
            api_key=_api_key(),
            device=self.launch_config.device or "cuda",
        )

    async def preload(self) -> None:
        if self._preload_future is not None:
            await asyncio.shield(self._preload_future)
            return
        loop = asyncio.get_running_loop()
        self._preload_future = loop.create_future()
        try:
            request = await self._run(
                self._build_request,
                prompt=self.entry.default_prompt or "",
                image=None,
                input_path="",
                video_path="",
            )
            result = await self._run(
                self.manager.run_realtime,
                entry=self.entry,
                request=request,
                action="configure",
            )
            self._accept_realtime_spec(result)
            if self.warmup_chunks and self.warmup_image_path and self.entry.supports_stream:
                await self._warmup()
            self._preload_future.set_result(None)
        except Exception as exc:
            self._preload_error = str(exc)
            self._preload_future.set_exception(exc)
            traceback.print_exc()
            raise

    async def _warmup(self) -> None:
        """Compile and stabilize the actual resident stream before user input."""

        def open_seed() -> Image.Image:
            with Image.open(self.warmup_image_path) as source:
                return source.convert("RGB")

        started = time.perf_counter()
        image = await self._run(open_seed)
        request = await self._run(
            self._build_request,
            prompt=self.entry.default_prompt or "",
            image=image,
            input_path=self.warmup_image_path,
            video_path="",
        )
        configured = await self._run(
            self.manager.run_realtime,
            entry=self.entry,
            request=request,
            action="configure",
        )
        self._accept_realtime_spec(configured)
        interactions = ["forward"]
        try:
            for index in range(self.warmup_chunks):
                stream_request = replace(
                    request,
                    image=request.image if index == 0 else None,
                    image_path=request.image_path if index == 0 else None,
                    interactions=interactions,
                    call_kwargs=_interactive_call_kwargs(
                        request.call_kwargs,
                        interactions,
                        seed=41_000 + index,
                    ),
                )
                result = await self._run(
                    self.manager.run_realtime,
                    entry=self.entry,
                    request=stream_request,
                    action="stream",
                )
                await self._run(realtime_frames_from_result, result)
        finally:
            await self._run(
                self.manager.run_realtime,
                entry=self.entry,
                request=request,
                action="reset",
            )
        self.warmup_ms = (time.perf_counter() - started) * 1000.0

    async def configure(
        self,
        *,
        prompt: str,
        image_path: str,
        video_path: str,
        dense_video_path: str = "",
        sparse_video_path: str = "",
    ) -> Image.Image:
        await self.preload()
        if self.queued_segment_generation:
            missing = [
                label
                for value, label in (
                    (prompt.strip(), "prompt"),
                    (image_path, "initial image"),
                    (dense_video_path, "dense depth control video"),
                    (sparse_video_path, "sparse pointmap/track control video"),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    "Queued segment setup is missing: " + ", ".join(missing) + "."
                )
        image: Image.Image | None = None
        if image_path:
            with Image.open(image_path) as source:
                image = source.convert("RGB")
        input_path = image_path or video_path
        request = await self._run(
            self._build_request,
            prompt=prompt,
            image=image,
            input_path=input_path,
            video_path=video_path,
            dense_video_path=dense_video_path,
            sparse_video_path=sparse_video_path,
        )
        configured = await self._run(
            self.manager.run_realtime,
            entry=self.entry,
            request=request,
            action="configure",
        )
        self._accept_realtime_spec(configured)
        self._base_request = request
        self._configured = True
        self._first_stream_step = True
        self._seed_image = image or Image.new("RGB", (1280, 720), "black")
        return self._seed_image.copy()

    def next_chunk_frames(self, default: int) -> int:
        """Return the cadence advertised by the resident model adapter."""

        del default
        if self._first_stream_step:
            return self.realtime_spec.first_chunk_frames
        return self.realtime_spec.steady_chunk_frames

    def steady_chunk_frames(self, default: int) -> int:
        del default
        return self.realtime_spec.steady_chunk_frames

    @property
    def supports_text_events(self) -> bool:
        """Whether this resident adapter can change text without resetting world state."""

        return bool(
            self.queued_segment_generation
            or "prompt_update" in self.realtime_spec.controls
            or self.entry.model_id in _PROMPT_BOUNDARY_MODEL_IDS
        )

    async def generate(
        self,
        interactions: list[str],
        *,
        seed: int,
        control_segments: list[ControlSegment] | None = None,
        prompt: str | None = None,
        dense_video_path: str | None = None,
        sparse_video_path: str | None = None,
    ) -> tuple[list[np.ndarray], float]:
        if not self._configured or self._base_request is None:
            raise RuntimeError("Realtime runtime is not configured.")
        if self.queued_segment_generation:
            call_kwargs = dict(self._base_request.call_kwargs)
            if not self._first_stream_step:
                if not dense_video_path or not sparse_video_path:
                    raise ValueError(
                        "Every EXTEND request needs a new dense depth video and sparse "
                        "pointmap/track video."
                    )
                call_kwargs["dense_video"] = dense_video_path
                call_kwargs["sparse_video"] = sparse_video_path
            request = replace(
                self._base_request,
                prompt=str(prompt).strip() if prompt is not None else self._base_request.prompt,
                interactions=[],
                call_kwargs=call_kwargs,
            )
            if not request.prompt:
                raise ValueError("Queued segment generation requires a non-empty prompt.")
            action = "run" if self._first_stream_step else "stream"
            if not self._first_stream_step:
                request = replace(
                    request,
                    input_path="",
                    image=None,
                    image_path=None,
                    video_path=None,
                )
        else:
            call_kwargs = _interactive_call_kwargs(
                self._base_request.call_kwargs,
                interactions,
                seed=seed,
                control_segments=control_segments,
            )
            request = replace(
                self._base_request,
                prompt=str(prompt).strip() if prompt is not None else self._base_request.prompt,
                interactions=list(interactions),
                call_kwargs=call_kwargs,
            )
            if not self._first_stream_step:
                request = replace(request, image=None, image_path=None)
            action = "stream"
        started = time.perf_counter()
        result = await self._run(
            self.manager.run_realtime,
            entry=self.entry,
            request=request,
            action=action,
        )
        self._accept_realtime_spec(result)
        self._first_stream_step = False
        self._base_request = request
        self.last_generation_metrics = (
            dict(result.get("realtime_metrics") or {})
            if isinstance(result, Mapping)
            else {}
        )
        frames = await self._run(realtime_frames_from_result, result)
        generation_ms = (time.perf_counter() - started) * 1000.0
        return frames, generation_ms

    async def reset(self) -> None:
        if self._base_request is not None:
            try:
                await self._run(
                    self.manager.run_realtime,
                    entry=self.entry,
                    request=self._base_request,
                    action="reset",
                )
            except Exception:
                pass
        self._configured = False
        self._base_request = None
        self._seed_image = None
        self._first_stream_step = True

    async def close(self) -> None:
        await self.reset()
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def ready(self) -> bool:
        return bool(self._preload_future and self._preload_future.done() and not self._preload_future.cancelled() and self._preload_future.exception() is None)

    @property
    def preload_error(self) -> str | None:
        return self._preload_error


@dataclass(slots=True)
class _ActivePeer:
    peer: Any
    channel: Any | None
    frames: LatestFrameBuffer
    resampler: RealtimeControlResampler
    generation_task: asyncio.Task[Any] | None = None
    liveness_task: asyncio.Task[Any] | None = None
    input_task: asyncio.Task[Any] | None = None
    input_messages: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    closed: bool = False
    first_action: asyncio.Event = field(default_factory=asyncio.Event)
    action_arrivals: deque[float] = field(default_factory=deque)
    chunk_index: int = 0
    seed: int = 42
    last_client_message_at: float = 0.0
    prompt_scheduled: bool = False
    initial_segment_pending: bool = False
    pending_prompt: str | None = None
    pending_prompt_dirty: bool = False
    pending_steps: int = 0
    base_prompt: str = ""
    text_events: list[dict[str, str]] = field(default_factory=list)
    catalog_revision: int = 0
    active_event_id: str | None = None
    text_events_supported: bool = True
    output_resolution: OutputResolutionState = field(default_factory=OutputResolutionState)


@dataclass(slots=True)
class _ActiveSocket:
    socket: Any
    resampler: RealtimeControlResampler
    frame_packets: asyncio.Queue[bytes]
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    generation_task: asyncio.Task[Any] | None = None
    sender_task: asyncio.Task[Any] | None = None
    liveness_task: asyncio.Task[Any] | None = None
    closed: bool = False
    first_action: asyncio.Event = field(default_factory=asyncio.Event)
    action_arrivals: deque[float] = field(default_factory=deque)
    chunk_index: int = 0
    seed: int = 42
    dropped_frames: int = 0
    last_client_message_at: float = 0.0
    prompt_scheduled: bool = False
    initial_segment_pending: bool = False
    pending_prompt: str | None = None
    pending_prompt_dirty: bool = False
    pending_steps: int = 0
    base_prompt: str = ""
    text_events: list[dict[str, str]] = field(default_factory=list)
    catalog_revision: int = 0
    active_event_id: str | None = None
    text_events_supported: bool = True
    output_resolution: OutputResolutionState = field(default_factory=OutputResolutionState)
    queued_segments: bool = False
    pending_segment: dict[str, str] | None = None
    segment_inflight: bool = False


class RealtimePeerManager:
    def __init__(
        self,
        *,
        runtime: ResidentWorldRuntime,
        fps: int,
        chunk_frames: int,
        ice_servers: list[dict[str, Any]] | None = None,
    ) -> None:
        self.runtime = runtime
        self.fps = fps
        self.chunk_frames = chunk_frames
        self.ice_servers = list(ice_servers or [])
        self._active: _ActivePeer | None = None
        self._active_socket: _ActiveSocket | None = None
        self._lock = asyncio.Lock()
        self._draining = False
        self._drain_done = asyncio.Event()
        self._drain_done.set()

    @staticmethod
    def _request_id(payload: Mapping[str, Any]) -> str | None:
        value = payload.get("request_id")
        return str(value)[:128] if value is not None else None

    @staticmethod
    def _event_for(active: _ActivePeer | _ActiveSocket, event_id: str) -> dict[str, str] | None:
        return next(
            (event for event in active.text_events if event["event_id"] == event_id),
            None,
        )

    @staticmethod
    def _schedule_prompt(
        active: _ActivePeer | _ActiveSocket,
        prompt: str,
        *,
        generate: bool,
    ) -> None:
        already_scheduled = active.pending_prompt_dirty
        active.pending_prompt = prompt
        active.pending_prompt_dirty = True
        if not generate or already_scheduled:
            return
        arrival = asyncio.get_running_loop().time()
        active.action_arrivals.append(arrival)
        active.first_action.set()

    def _handle_session_control(
        self,
        active: _ActivePeer | _ActiveSocket,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Apply an ordered control message and return its acknowledgement."""

        message_type = str(payload.get("type") or "").strip().lower()
        request_id = self._request_id(payload)
        if message_type == "event_catalog":
            if not active.text_events_supported:
                return {
                    "type": "event_catalog_ack",
                    "ok": False,
                    "request_id": request_id,
                    "catalog_revision": active.catalog_revision,
                    "message": "This model has no state-preserving realtime text update path.",
                }
            base_revision = payload.get("base_revision")
            if base_revision is not None:
                try:
                    revision_matches = int(base_revision) == active.catalog_revision
                except (TypeError, ValueError):
                    revision_matches = False
                if not revision_matches:
                    return {
                        "type": "event_catalog_ack",
                        "ok": False,
                        "request_id": request_id,
                        "catalog_revision": active.catalog_revision,
                        "event_catalog": active.text_events,
                        "active_event_id": active.active_event_id,
                        "message": "The event catalog changed; refresh before applying this edit.",
                    }
            try:
                events = normalize_text_events(payload.get("events"))
            except ValueError as exc:
                return {
                    "type": "event_catalog_ack",
                    "ok": False,
                    "request_id": request_id,
                    "catalog_revision": active.catalog_revision,
                    "message": str(exc),
                }
            previous = self._event_for(active, active.active_event_id or "")
            active.text_events = events
            active.catalog_revision += 1
            current = self._event_for(active, active.active_event_id or "")
            if active.active_event_id and current is None:
                active.active_event_id = None
                self._schedule_prompt(
                    active,
                    active.base_prompt,
                    generate=not getattr(active, "queued_segments", False),
                )
            elif current is not None and previous is not None and current["prompt"] != previous["prompt"]:
                self._schedule_prompt(
                    active,
                    current["prompt"],
                    generate=not getattr(active, "queued_segments", False),
                )
            return {
                "type": "event_catalog_ack",
                "ok": True,
                "request_id": request_id,
                "catalog_revision": active.catalog_revision,
                "event_catalog": active.text_events,
                "active_event_id": active.active_event_id,
                "status": "applied",
            }
        if message_type == "event":
            if not active.text_events_supported:
                return {
                    "type": "event_ack",
                    "ok": False,
                    "request_id": request_id,
                    "event_id": str(payload.get("event_id") or "") or None,
                    "state": str(payload.get("state") or "trigger"),
                    "active_event_id": active.active_event_id,
                    "message": "This model has no state-preserving realtime text update path.",
                }
            state = str(payload.get("state") or "trigger").strip().lower()
            if state in {"release", "off", "none"}:
                state = "clear"
            event_id = str(payload.get("event_id") or payload.get("id") or "").strip()
            if state not in {"trigger", "clear"}:
                return {
                    "type": "event_ack",
                    "ok": False,
                    "request_id": request_id,
                    "event_id": event_id or None,
                    "state": state,
                    "active_event_id": active.active_event_id,
                    "message": "event state must be 'trigger' or 'clear'.",
                }
            event = self._event_for(active, event_id) if state == "trigger" else None
            if state == "trigger" and event is None:
                return {
                    "type": "event_ack",
                    "ok": False,
                    "request_id": request_id,
                    "event_id": event_id or None,
                    "state": state,
                    "active_event_id": active.active_event_id,
                    "message": f"Unknown text event: {event_id or '<empty>'}.",
                }
            active.active_event_id = event_id if event is not None else None
            prompt = event["prompt"] if event is not None else active.base_prompt
            queued = bool(getattr(active, "queued_segments", False))
            self._schedule_prompt(active, prompt, generate=not queued)
            return {
                "type": "event_ack",
                "ok": True,
                "request_id": request_id,
                "event_id": event_id or None,
                "state": state,
                "active_event_id": active.active_event_id,
                "catalog_revision": active.catalog_revision,
                "applies_at": "next_segment" if queued else "next_chunk",
                "status": "accepted",
            }
        if message_type == "output_config":
            try:
                changed = active.output_resolution.update(payload.get("resolution"))
            except ValueError as exc:
                return {
                    "type": "output_config_ack",
                    "ok": False,
                    "request_id": request_id,
                    "resolution": active.output_resolution.to_payload(),
                    "resolution_revision": active.output_resolution.revision,
                    "message": str(exc),
                }
            return {
                "type": "output_config_ack",
                "ok": True,
                "request_id": request_id,
                "resolution": active.output_resolution.to_payload(),
                "resolution_revision": active.output_resolution.revision,
                "status": "queued" if changed else "unchanged",
                "applies_at": "next_chunk",
            }
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        if message_type == "step" or (
            message_type == "action" and str(action.get("event") or "").lower() == "step"
        ):
            if bool(getattr(active, "queued_segments", False)):
                return {
                    "type": "step_ack",
                    "ok": False,
                    "request_id": request_id,
                    "message": "Queued segment models require their next control videos.",
                }
            active.pending_steps += 1
            arrival = asyncio.get_running_loop().time()
            active.action_arrivals.append(arrival)
            active.first_action.set()
            return {
                "type": "step_ack",
                "ok": True,
                "request_id": request_id,
                "queued_steps": active.pending_steps,
                "status": "accepted",
            }
        return None

    @property
    def active(self) -> bool:
        return bool(
            self._draining
            or
            (self._active and not self._active.closed)
            or (self._active_socket and not self._active_socket.closed)
        )

    @property
    def draining(self) -> bool:
        return self._draining

    async def close_active(self) -> None:
        async with self._lock:
            active = self._active
            self._active = None
            active_socket = self._active_socket
            self._active_socket = None
            has_session = not (
                (active is None or active.closed)
                and (active_socket is None or active_socket.closed)
            )
            if has_session:
                self._draining = True
                self._drain_done.clear()
            wait_for_drain = self._draining and not has_session
        if wait_for_drain:
            await self._drain_done.wait()
            return
        if not has_session:
            return
        current_task = asyncio.current_task()
        try:
            if active is not None and not active.closed:
                active.closed = True
                tasks = (active.generation_task, active.liveness_task, active.input_task)
                for task in tasks:
                    if task is not None and task is not current_task:
                        task.cancel()
                await asyncio.gather(
                    *(task for task in tasks if task is not None and task is not current_task),
                    return_exceptions=True,
                )
                active.frames.close()
                await active.peer.close()
            if active_socket is not None and not active_socket.closed:
                active_socket.closed = True
                tasks = (
                    active_socket.generation_task,
                    active_socket.sender_task,
                    active_socket.liveness_task,
                )
                for task in tasks:
                    if task is not None and task is not current_task:
                        task.cancel()
                await asyncio.gather(
                    *(task for task in tasks if task is not None and task is not current_task),
                    return_exceptions=True,
                )
                if not active_socket.socket.closed:
                    await active_socket.socket.close()
        finally:
            try:
                await self.runtime.reset()
            finally:
                async with self._lock:
                    self._draining = False
                    self._drain_done.set()

    async def create_answer(self, *, offer: Mapping[str, Any], session: Mapping[str, Any]) -> dict[str, str]:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

        if self.runtime.queued_segment_generation:
            raise RuntimeError(
                "Queued segment generation uses the same-origin WebSocket transport; "
                "WebRTC/ICE is intentionally bypassed."
            )
        async with self._lock:
            if self.active:
                raise RuntimeError("A realtime world session is already active.")

            base_prompt = str(session.get("prompt") or self.runtime.entry.default_prompt or "")
            text_events = normalize_text_events(
                session.get("text_events", session.get("event_catalog"))
            )
            output_resolution = _new_output_resolution_state(
                self.runtime.entry,
                session.get("output_resolution"),
            )
            await self.runtime.configure(
                prompt=base_prompt,
                image_path=str(session.get("init_image_path") or ""),
                video_path=str(session.get("init_video_path") or ""),
                dense_video_path=str(session.get("dense_video_path") or ""),
                sparse_video_path=str(session.get("sparse_video_path") or ""),
            )
            self.fps = self.runtime.realtime_spec.fps
            frames = LatestFrameBuffer(
                maxsize=_env_int(
                    "WORLDFOUNDRY_REALTIME_FRAME_QUEUE",
                    max(self.runtime.steady_chunk_frames(self.chunk_frames), 8),
                ),
                backpressure_ms=_env_int(
                    "WORLDFOUNDRY_REALTIME_BACKPRESSURE_MS",
                    max(
                        int(self.runtime.steady_chunk_frames(self.chunk_frames) * 500 / self.fps),
                        50,
                    ),
                ),
            )
            rtc_configuration = RTCConfiguration(
                iceServers=[
                    RTCIceServer(
                        urls=server["urls"],
                        username=server.get("username"),
                        credential=server.get("credential"),
                    )
                    for server in self.ice_servers
                ]
            )
            pc = RTCPeerConnection(rtc_configuration)
            track = _build_video_track(
                frames=frames,
                fps=self.fps,
            )
            pc.addTrack(track)
            loop = asyncio.get_running_loop()
            active = _ActivePeer(
                peer=pc,
                channel=None,
                frames=frames,
                resampler=RealtimeControlResampler(fps=self.fps, start_time=loop.time()),
                last_client_message_at=loop.time(),
                prompt_scheduled="prompt_update" in self.runtime.realtime_spec.controls,
                initial_segment_pending=(
                    "prompt_update" in self.runtime.realtime_spec.controls
                ),
                base_prompt=base_prompt,
                text_events=text_events,
                catalog_revision=1 if text_events else 0,
                text_events_supported=self.runtime.supports_text_events,
                output_resolution=output_resolution,
            )
            self._active = active

            @pc.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                active.channel = channel
                channel_open_time = asyncio.get_running_loop().time()
                active.last_client_message_at = channel_open_time
                active.resampler.reset(start_time=channel_open_time)

                @channel.on("message")
                def on_message(raw: Any) -> None:
                    active.input_messages.put_nowait(raw)

                @channel.on("close")
                def on_close() -> None:
                    asyncio.create_task(self.close_active())

                active.generation_task = asyncio.create_task(self._generation_worker(active))
                active.liveness_task = asyncio.create_task(self._liveness_watchdog(active))
                active.input_task = asyncio.create_task(self._input_worker(active))
                if active.prompt_scheduled:
                    # Start the first native AR segment immediately. Subsequent
                    # segments are triggered by explicit prompt updates.
                    active.first_action.set()
                self._send(
                    channel,
                    {
                        "type": "ready",
                        "fps": self.fps,
                        "event_catalog": active.text_events,
                        "catalog_revision": active.catalog_revision,
                        "active_event_id": active.active_event_id,
                        "resolution": active.output_resolution.to_payload(),
                        "resolution_revision": active.output_resolution.revision,
                    },
                )

            @pc.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                if pc.connectionState in {"failed", "closed", "disconnected"}:
                    await self.close_active()

            try:
                await pc.setRemoteDescription(
                    RTCSessionDescription(sdp=str(offer["sdp"]), type=str(offer["type"]))
                )
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await _wait_for_ice_gathering(pc)
                return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
            except Exception:
                active.closed = True
                self._active = None
                active.frames.close()
                await pc.close()
                await self.runtime.reset()
                raise

    async def serve_socket(self, socket: Any) -> None:
        """Run a same-origin fallback session when UDP ICE cannot cross a tunnel."""

        from aiohttp import WSMsgType

        first = await asyncio.wait_for(
            socket.receive(),
            timeout=_env_int("WORLDFOUNDRY_REALTIME_SOCKET_SETUP_TIMEOUT_SECONDS", 15),
        )
        if first.type != WSMsgType.TEXT:
            raise RuntimeError("Expected a WebSocket configure message.")
        try:
            payload = json.loads(first.data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("WebSocket configure message is not valid JSON.") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("WebSocket configure message must be a JSON object.")
        if str(payload.get("type") or "").lower() != "configure":
            raise RuntimeError("Expected WebSocket message type='configure'.")
        session = payload.get("session") if isinstance(payload.get("session"), Mapping) else {}
        base_prompt = str(session.get("prompt") or self.runtime.entry.default_prompt or "")
        text_events = normalize_text_events(
            session.get("text_events", session.get("event_catalog"))
        )
        output_resolution = _new_output_resolution_state(
            self.runtime.entry,
            session.get("output_resolution"),
        )
        await socket.send_str(
            json.dumps(
                {"type": "configuring", "transport": "ws"},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        async with self._lock:
            if self.active:
                raise RuntimeError("A realtime world session is already active.")
            await self.runtime.configure(
                prompt=base_prompt,
                image_path=str(session.get("init_image_path") or ""),
                video_path=str(session.get("init_video_path") or ""),
                dense_video_path=str(session.get("dense_video_path") or ""),
                sparse_video_path=str(session.get("sparse_video_path") or ""),
            )
            self.fps = self.runtime.realtime_spec.fps
            loop = asyncio.get_running_loop()
            active = _ActiveSocket(
                socket=socket,
                resampler=RealtimeControlResampler(fps=self.fps, start_time=loop.time()),
                frame_packets=asyncio.Queue(
                    maxsize=_env_int(
                        "WORLDFOUNDRY_REALTIME_SOCKET_FRAME_QUEUE",
                        max(self.runtime.steady_chunk_frames(self.chunk_frames) * 2, 16),
                    )
                ),
                last_client_message_at=loop.time(),
                prompt_scheduled="prompt_update" in self.runtime.realtime_spec.controls,
                initial_segment_pending=(
                    "prompt_update" in self.runtime.realtime_spec.controls
                ),
                base_prompt=base_prompt,
                text_events=text_events,
                catalog_revision=1 if text_events else 0,
                text_events_supported=self.runtime.supports_text_events,
                output_resolution=output_resolution,
                queued_segments=self.runtime.queued_segment_generation,
            )
            self._active_socket = active

        active.generation_task = asyncio.create_task(
            self._socket_generation_worker(active),
            name="world-realtime-socket-generation",
        )
        active.sender_task = asyncio.create_task(
            self._socket_sender(active),
            name="world-realtime-socket-sender",
        )
        # aiohttp's protocol heartbeat detects dead sockets without depending
        # on JavaScript timers, which browsers throttle aggressively in
        # background tabs during long segment inference.
        active.liveness_task = None
        await self._socket_send(
            active,
            {
                "type": "ready",
                "fps": self.fps,
                "transport": "ws",
                "event_catalog": active.text_events,
                "catalog_revision": active.catalog_revision,
                "active_event_id": active.active_event_id,
                "resolution": active.output_resolution.to_payload(),
                "resolution_revision": active.output_resolution.revision,
            },
        )
        # Publish readiness before scheduling the first segment so the browser
        # cannot have a GENERATING status overwritten by a late ready event.
        if active.prompt_scheduled or active.queued_segments:
            active.first_action.set()

        try:
            async for message in socket:
                if message.type == WSMsgType.TEXT:
                    await self._handle_socket_message(active, message.data)
                elif message.type in {WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR}:
                    break
        finally:
            if self._active_socket is active:
                await self.close_active()

    async def _handle_socket_message(self, active: _ActiveSocket, raw: str) -> None:
        if active.closed:
            return
        active.last_client_message_at = asyncio.get_running_loop().time()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, Mapping):
            return
        message_type = str(payload.get("type") or "").lower()
        if message_type == "disconnect":
            await self.close_active()
            return
        if message_type in {"heartbeat", "ping"}:
            return
        acknowledgement = self._handle_session_control(active, payload)
        if acknowledgement is not None:
            await self._socket_send(active, acknowledgement)
            return
        if message_type == "prompt_update" and active.prompt_scheduled:
            prompt = str(payload.get("prompt") or "").strip()
            if prompt:
                arrival = asyncio.get_running_loop().time()
                active.base_prompt = prompt
                active.active_event_id = None
                active.pending_prompt = prompt
                active.pending_prompt_dirty = True
                active.action_arrivals.append(arrival)
                active.first_action.set()
            return
        if message_type == "segment_update" and active.queued_segments:
            prompt = str(payload.get("prompt") or "").strip()
            selected_event = self._event_for(active, active.active_event_id or "")
            if selected_event is not None:
                prompt = selected_event["prompt"]
            dense_video_path = str(payload.get("dense_video_path") or "").strip()
            sparse_video_path = str(payload.get("sparse_video_path") or "").strip()
            if not prompt or not dense_video_path or not sparse_video_path:
                await self._socket_send(
                    active,
                    {
                        "type": "segment_rejected",
                        "message": "EXTEND requires a prompt plus new dense and sparse control videos.",
                    },
                )
                return
            if active.segment_inflight or active.first_action.is_set() or active.pending_segment:
                await self._socket_send(
                    active,
                    {
                        "type": "segment_rejected",
                        "message": "A segment is already generating or queued.",
                    },
                )
                return
            expected_frames = self.runtime.realtime_spec.steady_chunk_frames
            try:
                await asyncio.gather(
                    asyncio.to_thread(
                        _validate_control_video_frames,
                        dense_video_path,
                        expected_frames=expected_frames,
                    ),
                    asyncio.to_thread(
                        _validate_control_video_frames,
                        sparse_video_path,
                        expected_frames=expected_frames,
                    ),
                )
            except ValueError as exc:
                await self._socket_send(
                    active,
                    {"type": "segment_rejected", "message": str(exc)},
                )
                return
            arrival = asyncio.get_running_loop().time()
            active.pending_segment = {
                "prompt": prompt,
                "dense_video_path": dense_video_path,
                "sparse_video_path": sparse_video_path,
            }
            active.action_arrivals.append(arrival)
            active.first_action.set()
            await self._socket_send(
                active,
                {"type": "segment_queued", "segment_index": active.chunk_index + 1},
            )
            return
        if message_type != "action":
            return
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        event = str(action.get("event") or "").lower()
        key = str(action.get("key") or "")
        arrival = asyncio.get_running_loop().time()
        if active.resampler.on_edge(arrival_time=arrival, event=event, key=key):
            active.action_arrivals.append(arrival)
            active.first_action.set()

    async def _socket_generation_worker(self, active: _ActiveSocket) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not active.closed:
                await active.first_action.wait()
                await asyncio.sleep(0)
                next_prompt = None
                prompt_changed = False
                prompt_event_id: str | None = None
                forced_step = False
                segments = None
                dense_video_path = None
                sparse_video_path = None
                if active.queued_segments:
                    active.first_action.clear()
                    active.segment_inflight = True
                    pending_segment = active.pending_segment
                    active.pending_segment = None
                    if pending_segment is not None:
                        next_prompt = pending_segment["prompt"]
                        dense_video_path = pending_segment["dense_video_path"]
                        sparse_video_path = pending_segment["sparse_video_path"]
                    active.pending_prompt = None
                    active.pending_prompt_dirty = False
                    interactions = []
                    await self._socket_send(
                        active,
                        {
                            "type": "segment_started",
                            "segment_index": active.chunk_index + 1,
                        },
                    )
                elif active.prompt_scheduled:
                    # The automatic first segment, each prompt boundary, and
                    # every acknowledged STEP are distinct work items. A STEP
                    # must never disappear into the automatic first segment.
                    active.first_action.clear()
                    active.segment_inflight = True
                    if active.initial_segment_pending:
                        active.initial_segment_pending = False
                        # A prompt received before inference begins still owns
                        # the next boundary, but an explicit STEP stays queued.
                        prompt_changed = active.pending_prompt_dirty
                    elif active.pending_prompt_dirty:
                        prompt_changed = True
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                    else:
                        active.segment_inflight = False
                        active.first_action.clear()
                        continue
                    if prompt_changed:
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                    interactions: list[str] = []
                    await self._socket_send(
                        active,
                        {
                            "type": "segment_started",
                            "segment_index": active.chunk_index + 1,
                        },
                    )
                else:
                    # Prompt changes may share the next native control chunk,
                    # while STEP remains a separate, action-free work item.
                    if active.pending_prompt_dirty:
                        prompt_changed = True
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                        frame_budget = self.runtime.next_chunk_frames(self.chunk_frames)
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                        interactions = []
                    else:
                        frame_budget = self.runtime.next_chunk_frames(self.chunk_frames)
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    if not interactions and not forced_step and not prompt_changed:
                        active.first_action.clear()
                        continue
                action_arrival = active.action_arrivals[0] if active.action_arrivals else None
                generation_started = loop.time()
                generation_task = asyncio.create_task(
                    self.runtime.generate(
                        interactions,
                        seed=active.seed + active.chunk_index,
                        control_segments=segments,
                        prompt=next_prompt,
                        dense_video_path=dense_video_path,
                        sparse_video_path=sparse_video_path,
                    ),
                    name="world-realtime-segment-inference",
                )
                try:
                    if active.queued_segments or active.prompt_scheduled:
                        interval = _env_int("WORLDFOUNDRY_SEGMENT_PROGRESS_SECONDS", 1)
                        while not generation_task.done():
                            done, _ = await asyncio.wait({generation_task}, timeout=interval)
                            if done:
                                break
                            await self._socket_send(
                                active,
                                {
                                    "type": "segment_progress",
                                    "segment_index": active.chunk_index + 1,
                                    "elapsed_ms": round(
                                        (loop.time() - generation_started) * 1000.0,
                                        1,
                                    ),
                                },
                            )
                    frames, generation_ms = await generation_task
                except ValueError as exc:
                    if not active.queued_segments:
                        raise
                    active.segment_inflight = False
                    await self._socket_send(
                        active,
                        {"type": "segment_rejected", "message": str(exc)},
                    )
                    continue
                except BaseException:
                    generation_task.cancel()
                    await asyncio.gather(generation_task, return_exceptions=True)
                    raise
                encode_started = loop.time()
                quality = _env_int(
                    "WORLDFOUNDRY_QUEUED_SEGMENT_JPEG_QUALITY"
                    if active.queued_segments
                    else "WORLDFOUNDRY_REALTIME_SOCKET_JPEG_QUALITY",
                    95 if active.queued_segments else 88,
                )
                while True:
                    active.output_resolution.observe_source(frames[0] if frames else None)
                    resolution_snapshot = active.output_resolution.snapshot()
                    packets = await asyncio.to_thread(
                        _encode_jpeg_frames,
                        frames,
                        quality=quality,
                        subsampling=0 if active.queued_segments else 1,
                        output_resolution=resolution_snapshot.dimensions,
                    )
                    if resolution_snapshot.revision == active.output_resolution.revision:
                        break
                    # A config ACK raced this encode. Never enqueue obsolete
                    # packets after that ACK; re-encode the same RGB chunk at
                    # the newest boundary without rerunning model inference.
                    active.dropped_frames += len(packets)
                accepted = 0
                for packet in packets:
                    if active.queued_segments or active.prompt_scheduled:
                        # Segment playback is count-exact: once chunk_done
                        # announces N frames, the browser must receive all N or
                        # it can never finish PLAYING. Backpressure this rare,
                        # explicit-boundary path instead of evicting frames.
                        await active.frame_packets.put(packet)
                        accepted += 1
                        continue
                    while active.frame_packets.full():
                        try:
                            active.frame_packets.get_nowait()
                            active.dropped_frames += 1
                        except asyncio.QueueEmpty:
                            break
                    active.frame_packets.put_nowait(packet)
                    accepted += 1
                enqueue_ms = (loop.time() - encode_started) * 1000.0
                active.chunk_index += 1
                now = loop.time()
                while active.action_arrivals and active.action_arrivals[0] <= generation_started:
                    active.action_arrivals.popleft()
                control_latency_ms = (
                    (now - action_arrival) * 1000.0 if action_arrival is not None else None
                )
                await self._socket_send(
                    active,
                    {
                        "type": "chunk_done",
                        "chunk_index": active.chunk_index,
                        "frames": accepted,
                        "generation_ms": round(generation_ms, 1),
                        "enqueue_ms": round(enqueue_ms, 1),
                        "control_latency_ms": round(control_latency_ms, 1)
                        if control_latency_ms is not None
                        else None,
                        "queue_depth": active.frame_packets.qsize(),
                        "dropped_frames": active.dropped_frames,
                        "resolution": resolution_snapshot.to_payload(),
                        "resolution_revision": resolution_snapshot.revision,
                        "interactions": (
                            ["queued_segment"]
                            if active.queued_segments
                            else [f"text_event:{prompt_event_id}"]
                            if prompt_event_id
                            else ["prompt_update"]
                            if prompt_changed
                            else ["step"]
                            if forced_step and not interactions
                            else interactions
                        ),
                        **{
                            key: round(float(value), 1)
                            for key, value in self.runtime.last_generation_metrics.items()
                            if key in {"condition_ms", "model_ms", "decode_ms"}
                        },
                    },
                )
                # The queue owns each encoded packet now. Do not retain a
                # second full segment of RGB/JPEG data while waiting for the
                # user's next boundary command.
                del packets, frames
                active.segment_inflight = False
                if active.queued_segments:
                    continue
                if (
                    active.initial_segment_pending
                    or active.pending_prompt_dirty
                    or active.pending_steps
                    or active.resampler.effective_keys
                ):
                    active.first_action.set()
                else:
                    active.first_action.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._socket_send(
                active,
                {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
            )
            await self.close_active()

    async def _socket_sender(self, active: _ActiveSocket) -> None:
        loop = asyncio.get_running_loop()
        deadline: float | None = None
        try:
            while not active.closed:
                packet = await active.frame_packets.get()
                now = loop.time()
                if deadline is not None:
                    deadline += 1.0 / self.fps
                    if deadline > now:
                        await asyncio.sleep(deadline - now)
                    elif now - deadline > 1.0 / self.fps:
                        deadline = now
                else:
                    deadline = now
                async with active.send_lock:
                    await active.socket.send_bytes(packet)
        except asyncio.CancelledError:
            raise
        except Exception:
            await self.close_active()

    async def _socket_liveness_watchdog(self, active: _ActiveSocket) -> None:
        interval = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_INTERVAL_SECONDS", 1)
        timeout = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_TIMEOUT_SECONDS", 30)
        try:
            while not active.closed:
                await asyncio.sleep(interval)
                if asyncio.get_running_loop().time() - active.last_client_message_at > timeout:
                    await self.close_active()
                    return
        except asyncio.CancelledError:
            raise

    @staticmethod
    async def _socket_send(active: _ActiveSocket, payload: Mapping[str, Any]) -> None:
        if active.closed or active.socket.closed:
            return
        async with active.send_lock:
            await active.socket.send_str(
                json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"))
            )

    async def _input_worker(self, active: _ActivePeer) -> None:
        """Apply reliable DataChannel messages in their original order."""

        try:
            while not active.closed:
                raw = await active.input_messages.get()
                await self._handle_message(active, raw)
        except asyncio.CancelledError:
            raise

    async def _handle_message(self, active: _ActivePeer, raw: Any) -> None:
        if active.closed or not isinstance(raw, str):
            return
        active.last_client_message_at = asyncio.get_running_loop().time()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, Mapping):
            return
        message_type = str(payload.get("type") or "").lower()
        if message_type == "disconnect":
            await self.close_active()
            return
        if message_type in {"heartbeat", "ping"}:
            return
        acknowledgement = self._handle_session_control(active, payload)
        if acknowledgement is not None:
            self._send(active.channel, acknowledgement)
            return
        if message_type == "prompt_update" and active.prompt_scheduled:
            prompt = str(payload.get("prompt") or "").strip()
            if prompt:
                arrival = asyncio.get_running_loop().time()
                active.base_prompt = prompt
                active.active_event_id = None
                active.pending_prompt = prompt
                active.pending_prompt_dirty = True
                active.action_arrivals.append(arrival)
                active.first_action.set()
            return
        if message_type != "action":
            return
        action = payload.get("action") if isinstance(payload.get("action"), Mapping) else {}
        event = str(action.get("event") or "").lower()
        key = str(action.get("key") or "")
        arrival = asyncio.get_running_loop().time()
        if active.resampler.on_edge(arrival_time=arrival, event=event, key=key):
            active.action_arrivals.append(arrival)
            active.first_action.set()

    async def _liveness_watchdog(self, active: _ActivePeer) -> None:
        interval = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_INTERVAL_SECONDS", 1)
        timeout = _env_int("WORLDFOUNDRY_REALTIME_LIVENESS_TIMEOUT_SECONDS", 30)
        try:
            while not active.closed:
                await asyncio.sleep(interval)
                if asyncio.get_running_loop().time() - active.last_client_message_at > timeout:
                    await self.close_active()
                    return
        except asyncio.CancelledError:
            raise

    async def _generation_worker(self, active: _ActivePeer) -> None:
        loop = asyncio.get_running_loop()
        try:
            while not active.closed:
                await active.first_action.wait()
                # Let the ordered input worker drain the current network burst
                # before sampling. This coalesces adjacent key edges (for
                # example a diagonal joystick gesture) without a timer-based
                # debounce or an extra frame of latency.
                await asyncio.sleep(0)
                next_prompt = None
                prompt_changed = False
                prompt_event_id: str | None = None
                forced_step = False
                segments = None
                if active.prompt_scheduled:
                    active.first_action.clear()
                    if active.initial_segment_pending:
                        active.initial_segment_pending = False
                        prompt_changed = active.pending_prompt_dirty
                    elif active.pending_prompt_dirty:
                        prompt_changed = True
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                    else:
                        active.first_action.clear()
                        continue
                    if prompt_changed:
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                    interactions: list[str] = []
                else:
                    if active.pending_prompt_dirty:
                        prompt_changed = True
                        prompt_event_id = active.active_event_id
                        next_prompt = active.pending_prompt
                        active.pending_prompt = None
                        active.pending_prompt_dirty = False
                        frame_budget = self.runtime.next_chunk_frames(self.chunk_frames)
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    elif active.pending_steps:
                        active.pending_steps -= 1
                        forced_step = True
                        interactions = []
                    else:
                        frame_budget = self.runtime.next_chunk_frames(self.chunk_frames)
                        segments = active.resampler.sample_chunk(
                            frame_budget,
                            wall_time=loop.time(),
                        )
                        interactions = interactions_from_segments(segments)
                    if not interactions and not forced_step and not prompt_changed:
                        active.first_action.clear()
                        continue
                action_arrival = active.action_arrivals[0] if active.action_arrivals else None
                generation_started = loop.time()
                frames, generation_ms = await self.runtime.generate(
                    interactions,
                    seed=active.seed + active.chunk_index,
                    control_segments=segments,
                    prompt=next_prompt,
                )
                while True:
                    active.output_resolution.observe_source(frames[0] if frames else None)
                    resolution_snapshot = active.output_resolution.snapshot()
                    if resolution_snapshot.dimensions is None or all(
                        (frame.shape[1], frame.shape[0])
                        == resolution_snapshot.dimensions
                        for frame in frames
                    ):
                        transport_frames = frames
                    else:
                        transport_frames = await asyncio.to_thread(
                            _resize_rgb_frames,
                            frames,
                            output_resolution=resolution_snapshot.dimensions,
                        )
                    if resolution_snapshot.revision == active.output_resolution.revision:
                        break
                accepted = await active.frames.put_chunk(transport_frames)
                active.chunk_index += 1
                now = loop.time()
                while active.action_arrivals and active.action_arrivals[0] <= generation_started:
                    active.action_arrivals.popleft()
                control_latency_ms = (
                    (now - action_arrival) * 1000.0 if action_arrival is not None else None
                )
                self._send(
                    active.channel,
                    {
                        "type": "chunk_done",
                        "chunk_index": active.chunk_index,
                        "frames": accepted,
                        "generation_ms": round(generation_ms, 1),
                        "enqueue_ms": round(active.frames.last_enqueue_ms, 1),
                        "control_latency_ms": round(control_latency_ms, 1)
                        if control_latency_ms is not None
                        else None,
                        "queue_depth": active.frames.qsize(),
                        "dropped_frames": active.frames.dropped_frames,
                        "resolution": resolution_snapshot.to_payload(),
                        "resolution_revision": resolution_snapshot.revision,
                        "interactions": (
                            [f"text_event:{prompt_event_id}"]
                            if prompt_event_id
                            else ["prompt_update"]
                            if prompt_changed
                            else ["step"]
                            if forced_step and not interactions
                            else interactions
                        ),
                        **{
                            key: round(float(value), 1)
                            for key, value in self.runtime.last_generation_metrics.items()
                            if key in {"condition_ms", "model_ms", "decode_ms"}
                        },
                    },
                )
                if (
                    active.initial_segment_pending
                    or active.pending_prompt_dirty
                    or active.pending_steps
                    or active.resampler.effective_keys
                ):
                    active.first_action.set()
                else:
                    active.first_action.clear()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._send(active.channel, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            await self.close_active()

    @staticmethod
    def _send(channel: Any | None, payload: Mapping[str, Any]) -> None:
        if channel is None or getattr(channel, "readyState", "") != "open":
            return
        channel.send(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")))


def _build_video_track(
    *,
    frames: LatestFrameBuffer,
    fps: int,
) -> Any:
    from aiortc import MediaStreamTrack
    from aiortc.mediastreams import MediaStreamError
    from av import VideoFrame

    class RealtimeVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self) -> None:
            super().__init__()
            self._pts = 0
            self._time_base = Fraction(1, fps)
            self._interval = 1.0 / fps
            self._deadline: float | None = None
            self._last_array: np.ndarray | None = None

        async def recv(self) -> VideoFrame:
            loop = asyncio.get_running_loop()
            if self._last_array is None:
                try:
                    self._last_array = await frames.get()
                except EOFError as exc:
                    raise MediaStreamError from exc
                self._deadline = loop.time()
            else:
                assert self._deadline is not None
                now = loop.time()
                self._deadline += self._interval
                if self._deadline > now:
                    await asyncio.sleep(self._deadline - now)
                elif now - self._deadline > self._interval:
                    self._deadline = now
                try:
                    self._last_array = frames.get_nowait()
                except asyncio.QueueEmpty:
                    # Keep a constant RTP clock even between generated chunks.
                    # Besides giving the browser a stable live stream, the
                    # repeated hold frame flushes the encoder's final source
                    # frame instead of leaving it buffered until the next user
                    # action arrives.
                    pass
                except EOFError as exc:
                    raise MediaStreamError from exc
            # Frames are already resized in the generation worker. Keep the
            # aiortc cadence path free of synchronous resize/canvas work; a
            # held frame is therefore repeated at zero additional scale cost.
            frame = VideoFrame.from_ndarray(self._last_array, format="rgb24")
            frame.pts = self._pts
            frame.time_base = self._time_base
            self._pts += 1
            return frame

    return RealtimeVideoTrack()


async def _wait_for_ice_gathering(peer: Any) -> None:
    if peer.iceGatheringState == "complete":
        return
    complete = asyncio.Event()

    @peer.on("icegatheringstatechange")
    def on_icegatheringstatechange() -> None:
        if peer.iceGatheringState == "complete":
            complete.set()

    if peer.iceGatheringState == "complete":
        complete.set()
    await asyncio.wait_for(
        complete.wait(),
        timeout=_env_int("WORLDFOUNDRY_REALTIME_ICE_TIMEOUT_SECONDS", 15),
    )


def _require_realtime_dependencies(*, require_rtc: bool, require_av: bool) -> None:
    missing: list[str] = []
    names = ["aiohttp"]
    if require_rtc:
        names.append("aiortc")
    if require_rtc or require_av:
        names.append("av")
    for name in names:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "Realtime World frontend requires "
            + ", ".join(missing)
            + ". Install `worldfoundry[studio_realtime]`."
        )


def _prefer_websocket_transport(
    *,
    prompt_scheduled: bool,
    queued_segments: bool,
    remote: str,
) -> bool:
    """Choose the transport without making segment UIs depend on ICE."""

    return prompt_scheduled or queued_segments or remote in {"127.0.0.1", "::1"}


def serve_realtime_world_frontend(
    *,
    entry: CatalogEntry,
    launch_config: StudioLaunchConfig,
    manager: StudioManager,
    host: str,
    port: int,
    demo_images: tuple[Path, ...],
    allowed_roots: tuple[Path, ...],
) -> None:
    """Serve the model-specific WebRTC frontend."""

    prompt_scheduled = "prompt-scheduled" in entry.tags
    queued_segments = "queued-segment-generation" in entry.tags
    websocket_only = os.getenv(
        "WORLDFOUNDRY_REALTIME_WEBSOCKET_ONLY", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    _require_realtime_dependencies(
        # Same-origin WebSocket streaming is fully self-contained and is the
        # reliable transport for SSH tunnels.  Keep aiortc optional when the
        # operator explicitly selects that path; the default WebRTC behavior
        # and dependency check remain unchanged.
        require_rtc=not (prompt_scheduled or queued_segments or websocket_only),
        require_av=queued_segments,
    )
    from aiohttp import web

    from worldfoundry.studio.visualization.backends.world import (
        _reference_image_label,
        world_favicon_svg,
        world_frontend_css,
        world_frontend_html,
        world_frontend_js,
    )

    if prompt_scheduled or queued_segments or "user-input-only" in entry.tags:
        # These modes must start from the user's own upload. Do not silently
        # expose packaged reference images as product examples or use them for
        # warmup.
        demo_images = ()

    fps = _env_int("WORLDFOUNDRY_REALTIME_FPS", 16)
    chunk_frames = _realtime_frame_budget(
        entry,
        _env_int(
            "WORLDFOUNDRY_REALTIME_CHUNK_FRAMES",
            _default_realtime_chunk_frames(entry),
        ),
    )
    ice_servers = _ice_server_payload()
    runtime = ResidentWorldRuntime(
        manager=manager,
        entry=entry,
        launch_config=launch_config,
        fps=fps,
        warmup_image_path=str(demo_images[0]) if demo_images else "",
        warmup_chunks=_env_int(
            "WORLDFOUNDRY_REALTIME_WARMUP_CHUNKS",
            0,
            minimum=0,
        ),
    )
    peers = RealtimePeerManager(
        runtime=runtime,
        fps=fps,
        chunk_frames=chunk_frames,
        ice_servers=ice_servers,
    )
    upload_root = Path(manager.workspace_root).expanduser().resolve() / "world_realtime_inputs"
    upload_root.mkdir(parents=True, exist_ok=True)
    owned_uploads: set[Path] = set()
    upload_cleanup_lock = asyncio.Lock()
    file_roots = tuple(
        dict.fromkeys(root.expanduser().resolve() for root in (*allowed_roots, upload_root))
    )

    app = web.Application(client_max_size=2 * 1024**3)
    preload_task: asyncio.Task[Any] | None = None

    async def cleanup_owned_uploads() -> None:
        async with upload_cleanup_lock:
            paths = tuple(owned_uploads)
            owned_uploads.clear()
        await asyncio.gather(
            *(asyncio.to_thread(path.unlink, missing_ok=True) for path in paths),
            return_exceptions=True,
        )

    async def index(_: Any) -> Any:
        return web.Response(text=world_frontend_html(entry, launch_config), content_type="text/html")

    async def css(_: Any) -> Any:
        return web.Response(text=world_frontend_css(), content_type="text/css")

    async def js(_: Any) -> Any:
        return web.Response(text=world_frontend_js(), content_type="application/javascript")

    async def favicon(_: Any) -> Any:
        return web.Response(text=world_favicon_svg(), content_type="image/svg+xml")

    async def session_info(request: Any) -> Any:
        remote = str(request.remote or "").split("%", 1)[0]
        # Segment models exchange a small ordered command at each generation
        # boundary, then stream RGB frames. They do not benefit from an RTC
        # control channel, while ICE discovery adds a failure-prone startup
        # round trip for SSH tunnels and reverse proxies.
        segment_transport = prompt_scheduled or queued_segments or websocket_only
        prefer_socket = websocket_only or _prefer_websocket_transport(
            prompt_scheduled=prompt_scheduled,
            queued_segments=queued_segments,
            remote=remote,
        )
        examples = [
            {
                "label": _reference_image_label(path, index),
                "path": str(path),
                "url": f"/api/file?path={quote(str(path), safe='')}",
            }
            for index, path in enumerate(demo_images[:9], start=1)
        ]
        return web.json_response(
            {
                "model_id": entry.model_id,
                "display_name": entry.display_name,
                "transport": "websocket" if segment_transport else "webrtc+websocket",
                "runtime_engine": "worldfoundry-resident",
                "performance_contract": (
                    "in-tree-full-quality-segments" if queued_segments else "in-tree-realtime"
                ),
                "websocket_fallback": True,
                "prefer_websocket": prefer_socket,
                "supports_video_input": not (prompt_scheduled or queued_segments),
                "interaction_mode": (
                    "queued-segments"
                    if queued_segments
                    else "prompt-scheduled" if prompt_scheduled else "controls"
                ),
                "fps": runtime.realtime_spec.fps,
                "chunk_frames": runtime.realtime_spec.first_chunk_frames,
                "steady_chunk_frames": runtime.realtime_spec.steady_chunk_frames,
                "controls": list(runtime.realtime_spec.controls),
                "realtime_spec": runtime.realtime_spec.to_payload(),
                "capabilities": {
                    "text_events": runtime.supports_text_events,
                    "runtime_event_catalog": True,
                    "event_ack": True,
                    "no_action_step": not queued_segments,
                    "output_resolution": True,
                    "draggable_panels": True,
                },
                "event_catalog": [],
                "active_event_id": None,
                "output_resolution": {"mode": "native"},
                "output_resolutions": _output_resolution_options(entry),
                "output_resolution_scope": "transport",
                "ice_servers": ice_servers,
                "examples": examples,
            }
        )

    async def runtime_status(_: Any) -> Any:
        return web.json_response(
            {
                "ready": runtime.ready,
                "loading": not runtime.ready and runtime.preload_error is None,
                "error": runtime.preload_error,
                "warmup_ms": round(runtime.warmup_ms, 1),
                "session_active": peers.active,
                "draining": peers.draining,
                "realtime_spec": runtime.realtime_spec.to_payload(),
            }
        )

    async def load_runtime(_: Any) -> Any:
        await runtime.preload()
        return web.json_response({"loaded": True, "ready": True})

    async def upload_input(request: Any) -> Any:
        reader = await request.multipart()
        field = await reader.next()
        if field is None or field.name != "file":
            raise web.HTTPBadRequest(reason="Expected multipart field named 'file'.")
        filename = Path(field.filename or "input.bin").name
        suffix = Path(filename).suffix[:16]
        target = upload_root / f"{uuid.uuid4().hex}{suffix}"
        handle = await asyncio.to_thread(target.open, "wb")
        try:
            while chunk := await field.read_chunk(size=1024 * 1024):
                await asyncio.to_thread(handle.write, chunk)
        finally:
            await asyncio.to_thread(handle.close)
        owned_uploads.add(target)
        return web.json_response(
            {"path": str(target), "url": f"/api/file?path={quote(str(target), safe='')}"}
        )

    async def file_response(request: Any) -> Any:
        raw = str(request.query.get("path") or "")
        path = Path(raw).expanduser().resolve()
        if not any(path == root or root in path.parents for root in file_roots):
            raise web.HTTPForbidden(reason="Path is outside allowed Studio roots.")
        if not path.is_file():
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers={"Cache-Control": "no-store"})

    async def offer(request: Any) -> Any:
        payload = await request.json()
        offer_payload = payload.get("offer") if isinstance(payload.get("offer"), Mapping) else payload
        session_payload = payload.get("session") if isinstance(payload.get("session"), Mapping) else {}
        if not offer_payload.get("sdp") or not offer_payload.get("type"):
            raise web.HTTPBadRequest(reason="Expected WebRTC offer sdp and type.")
        try:
            answer = await peers.create_answer(offer=offer_payload, session=session_payload)
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except RuntimeError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        return web.json_response(answer)

    async def websocket(request: Any) -> Any:
        socket = web.WebSocketResponse(autoping=True, heartbeat=10, max_msg_size=1024**2)
        await socket.prepare(request)
        try:
            await peers.serve_socket(socket)
        except Exception as exc:
            if not socket.closed:
                await socket.send_str(
                    json.dumps(
                        {"type": "error", "message": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                await socket.close(code=1011, message=b"realtime session failed")
        finally:
            await cleanup_owned_uploads()
        return socket

    async def reset_session(request: Any) -> Any:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        await peers.close_active()
        if not bool(payload.get("preserve_uploads")):
            await cleanup_owned_uploads()
        return web.json_response({"reset": True})

    async def on_startup(_: Any) -> None:
        # Overlap model residency work with input selection while keeping the
        # HTML server immediately reachable.
        nonlocal preload_task
        preload_task = asyncio.create_task(runtime.preload(), name="world-realtime-preload")

        def consume_preload_result(task: asyncio.Task[Any]) -> None:
            if not task.cancelled():
                task.exception()

        preload_task.add_done_callback(consume_preload_result)

        stale_after = _env_int("WORLDFOUNDRY_UPLOAD_STALE_SECONDS", 24 * 60 * 60)
        cutoff = time.time() - stale_after
        stale = [
            path
            for path in upload_root.iterdir()
            if path.is_file() and path.stat().st_mtime < cutoff
        ]
        await asyncio.gather(
            *(asyncio.to_thread(path.unlink, missing_ok=True) for path in stale),
            return_exceptions=True,
        )

    async def on_shutdown(_: Any) -> None:
        if preload_task is not None and not preload_task.done():
            preload_task.cancel()
            await asyncio.gather(preload_task, return_exceptions=True)
        await peers.close_active()
        await runtime.close()
        await cleanup_owned_uploads()

    app.router.add_get("/", index)
    app.router.add_get("/world.css", css)
    app.router.add_get("/world.js", js)
    app.router.add_get("/favicon.svg", favicon)
    app.router.add_get("/api/session", session_info)
    app.router.add_get("/api/runtime/status", runtime_status)
    app.router.add_post("/api/runtime/load", load_runtime)
    app.router.add_post("/api/session/input", upload_input)
    app.router.add_get("/api/file", file_response)
    app.router.add_get("/api/realtime/ws", websocket)
    app.router.add_post("/api/webrtc/offer", offer)
    app.router.add_post("/api/session/reset", reset_session)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)

    print(f"WorldFoundry realtime WebRTC UI: http://{host}:{port}", flush=True)
    web.run_app(app, host=host, port=port, print=None, handle_signals=True)


__all__ = [
    "LatestFrameBuffer",
    "OutputResolutionState",
    "RealtimeControlResampler",
    "RealtimeControlState",
    "ResidentWorldRuntime",
    "interactions_from_keys",
    "interactions_from_segments",
    "normalize_output_resolution",
    "normalize_text_events",
    "realtime_frames_from_result",
    "serve_realtime_world_frontend",
]
