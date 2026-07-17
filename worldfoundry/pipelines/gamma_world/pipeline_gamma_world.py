"""WorldFoundry pipeline for inference with Gamma-World."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import tempfile
from typing import Any

from worldfoundry.core.io import normalize_action_track, write_json
from worldfoundry.core.utils import compose_horizontal_views
from worldfoundry.pipelines.video_official.pipeline_official_video import OfficialVideoPipeline


class GammaWorldPipeline(OfficialVideoPipeline):
    """Generate synchronized multi-player world rollouts from images and actions."""

    MODEL_ID = "gamma-world"
    GENERATION_TYPE = "i2v"

    def __call__(
        self,
        prompt: str,
        images: Any = None,
        video: Any = None,
        output_path: str | Path | None = None,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> Any:
        del video
        target = Path(output_path or "tmp/pipeline_eval/gamma-world.mp4").expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        runtime_kwargs = dict(kwargs)
        eval_dir = runtime_kwargs.pop("eval_dir", None)
        temp_dir: Path | None = None
        if eval_dir is None and isinstance(images, (str, Path)):
            sample_dir = Path(images).expanduser()
            if sample_dir.is_dir() and (sample_dir / "first_frame.png").is_file():
                eval_dir = sample_dir
        if eval_dir is None:
            if images is None:
                raise ValueError("Gamma-World requires images or an eval_dir")
            temp_dir = Path(tempfile.mkdtemp(prefix=".gamma_world_", dir=target.parent))
            num_frames = int(runtime_kwargs.get("num_frames", 189))
            width = int(runtime_kwargs.get("width", 480))
            height = int(runtime_kwargs.get("height", 320))
            is_view_sequence = (
                isinstance(images, Sequence)
                and not isinstance(images, (str, bytes, bytearray))
                and not hasattr(images, "shape")
            )
            views = list(images) if is_view_sequence else [images]
            n_players = int(
                runtime_kwargs.get("n_players")
                or (len(views) if is_view_sequence else 2)
            )
            runtime_kwargs["n_players"] = n_players
            target_size = (width, height) if len(views) > 1 else None
            compose_horizontal_views(views, target_size=target_size).save(
                temp_dir / "first_frame.png"
            )
            (temp_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            action_sources = runtime_kwargs.pop(
                "actions", runtime_kwargs.pop("action_paths", None)
            )
            if action_sources is not None:
                if isinstance(action_sources, (str, Path, Mapping)):
                    action_sources = [action_sources]
                if len(action_sources) != n_players:
                    raise ValueError(
                        f"received {len(action_sources)} action tracks for n_players={n_players}"
                    )
                for index, source in enumerate(action_sources):
                    write_json(
                        temp_dir / f"action_{index}.json",
                        normalize_action_track(source, num_frames=num_frames),
                    )
            eval_dir = temp_dir
        try:
            result = self.runtime.generate(
                prompt=prompt,
                output_path=target,
                eval_dir=str(Path(eval_dir).expanduser().resolve()),
                **runtime_kwargs,
            )
        finally:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
        if return_dict:
            return result
        if isinstance(result, Mapping) and result.get("status") == "failed":
            raise RuntimeError(str(result.get("error") or "Gamma-World inference failed"))
        return result.get("artifact_path") if isinstance(result, Mapping) else result


__all__ = ["GammaWorldPipeline"]
