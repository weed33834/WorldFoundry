"""Persistent low-latency rollout for LingBot-World-V2.

V2 shares the in-tree causal Wan session machinery with LingBot-World Fast,
but advertises its own four-latent cadence, attention window, denoising
schedule, and camera normalization.  No subprocess, MP4, or checkpoint reload
is allowed after the session has been prepared.
"""

from __future__ import annotations

from worldfoundry.synthesis.visual_generation.lingbot_world.realtime import (
    LingBotRealtimeSession,
)


class LingBotWorldV2RealtimeSession(LingBotRealtimeSession):
    """One persistent four-latent LingBot-World-V2 causal rollout."""

    chunk_latent_frames = 4
    first_output_frames = 13
    steady_output_frames = 16
    temporal_window_frames = 18
    sink_frames = 6
    cache_frames = 18
    timestep_indices = (0, 250, 500, 750)
    camera_move_speed_per_second = 1.6
    camera_rotate_speed_degrees_per_second = 3.2
    normalize_camera_translation = True
    compile_namespace = "lingbot-world-v2"
    compile_env = "WORLDFOUNDRY_LINGBOT_V2_REALTIME_COMPILE"
    condition_prefetch_chunks = 8
    condition_replenish_chunks = 4


__all__ = ["LingBotWorldV2RealtimeSession"]
