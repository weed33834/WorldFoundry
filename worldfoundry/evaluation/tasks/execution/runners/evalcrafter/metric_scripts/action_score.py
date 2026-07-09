"""EvalCrafter action score using the in-tree VideoMAE recipe.

WorldFoundry keeps the concrete VideoMAE config and K400 label map in
``base_models``. The MMAction2 framework itself remains an environment
dependency because EvalCrafter only needs its public inference API.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from operator import itemgetter
from pathlib import Path

import torch
from transformers import AutoTokenizer, CLIPModel

from worldfoundry.base_models.perception_core.action_recognition.videomae_mmaction import (
    checkpoint_path,
    config_path,
    label_map_path,
)
from worldfoundry.evaluation.tasks.execution.framework.benchmark_assets import bundled_benchmark_asset


def _runtime_root() -> Path:
    root = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_RUNTIME_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return Path.cwd().resolve()


def _checkpoints_dir(runtime_root: Path) -> Path:
    root = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR")
    return Path(root).expanduser().resolve() if root else runtime_root / "checkpoints"


def _metadata_path(runtime_root: Path) -> Path:
    explicit = os.environ.get("WORLDFOUNDRY_EVALCRAFTER_METADATA")
    if explicit:
        return Path(explicit).expanduser().resolve()
    bundled = bundled_benchmark_asset("evalcrafter", "metadata.json")
    return bundled if bundled.is_file() else runtime_root / "metadata.json"


def _clip_model_id(checkpoints_dir: Path) -> str:
    local = checkpoints_dir / "clip-vit-base-patch32"
    return str(local) if local.exists() else "openai/clip-vit-base-patch32"


def _load_action_runtime():
    try:
        from mmengine import Config
        from mmaction.apis import inference_recognizer, init_recognizer
    except ImportError as exc:
        raise RuntimeError(
            "EvalCrafter action_score requires the optional MMAction2 runtime. "
            "Install the benchmark environment profile before running official scoring."
        ) from exc
    return Config, inference_recognizer, init_recognizer


def _configure_logger(results_dir: Path, metric: str) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = results_dir / f"{metric}_record.txt"
    if log_file_path.exists():
        log_file_path.unlink()

    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(filename=str(log_file_path))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def _load_action_metadata(metadata_path: Path) -> dict[str, str]:
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    action_by_id: dict[str, str] = {}
    for item_key, item_value in data.items():
        action = item_value.get("attributes", {}).get("action", "")
        if action:
            action_by_id[item_key] = action
    return action_by_id


def _calculate_action_score(
    *,
    video_path: Path,
    action: str,
    action_model,
    clip_model: CLIPModel,
    clip_tokenizer: AutoTokenizer,
    labels: list[str],
    inference_recognizer,
    device: str,
) -> float:
    pred_result = inference_recognizer(action_model, str(video_path))
    pred_scores = pred_result.pred_scores.item.tolist()
    score_sorted = sorted(tuple(zip(range(len(pred_scores)), pred_scores)), key=itemgetter(1), reverse=True)
    top5_label = score_sorted[:5]

    results = [(labels[index], score) for index, score in top5_label]
    action_pred = [label for label, _score in results]
    confidence = [float(score) for _label, score in results]

    action_pred_tokens = clip_tokenizer(action_pred, return_tensors="pt", padding=True, truncation=True)
    text_tokens = clip_tokenizer(action, return_tensors="pt", padding=True, truncation=True)
    action_pred_input = action_pred_tokens["input_ids"].to(device)
    text_input = text_tokens["input_ids"].to(device)

    with torch.no_grad():
        action_pred_features = clip_model.get_text_features(action_pred_input)
        text_features = clip_model.get_text_features(text_input)

    action_pred_features = action_pred_features / action_pred_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    action_recog_similarities = action_pred_features @ text_features.T
    score = torch.tensor(confidence, device=device).unsqueeze(0) @ action_recog_similarities
    return float(score[0][0].detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_videos", type=str, required=True, help="Directory containing EvalCrafter videos")
    parser.add_argument("--metric", type=str, default="action_score", help="Metric name")
    args = parser.parse_args()

    runtime_root = _runtime_root()
    results_dir = Path(os.environ.get("WORLDFOUNDRY_EVALCRAFTER_RESULTS_DIR", runtime_root / "results"))
    checkpoints_dir = _checkpoints_dir(runtime_root)
    metadata_path = _metadata_path(runtime_root)
    videos_dir = Path(args.dir_videos).expanduser().resolve()

    _configure_logger(results_dir, args.metric)

    labels = [line.strip() for line in label_map_path().read_text(encoding="utf-8").splitlines() if line.strip()]
    action_by_id = _load_action_metadata(metadata_path)
    video_paths = sorted(path for path in videos_dir.iterdir() if path.suffix.lower() == ".mp4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Config, inference_recognizer, init_recognizer = _load_action_runtime()

    clip_model_id = _clip_model_id(checkpoints_dir)
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(device)
    clip_tokenizer = AutoTokenizer.from_pretrained(clip_model_id)

    video_mae_checkpoint = checkpoint_path(checkpoint_dir=checkpoints_dir / "VideoMAE")
    if not video_mae_checkpoint.is_file():
        raise FileNotFoundError(
            "Missing VideoMAE K400 checkpoint for EvalCrafter action_score: "
            f"{video_mae_checkpoint}. Set WORLDFOUNDRY_VIDEOMAE_K400_CKPT or stage it under "
            "WORLDFOUNDRY_EVALCRAFTER_CHECKPOINTS_DIR/VideoMAE."
        )

    cfg = Config.fromfile(str(config_path()))
    action_model = init_recognizer(cfg, str(video_mae_checkpoint), device=device)

    scores: list[float] = []
    for video_path in video_paths:
        video_id = video_path.stem[:4]
        action = action_by_id.get(video_id)
        if not action:
            continue
        score = _calculate_action_score(
            video_path=video_path,
            action=action,
            action_model=action_model,
            clip_model=clip_model,
            clip_tokenizer=clip_tokenizer,
            labels=labels,
            inference_recognizer=inference_recognizer,
            device=device,
        )
        scores.append(score)
        logging.info(
            "Vid: %s,  Current %s: %s, Current avg. %s: %s",
            video_id,
            args.metric,
            score,
            args.metric,
            sum(scores) / len(scores),
        )

    average_score = sum(scores) / len(scores) if scores else 0.0
    logging.info("Final average %s: %s, Total videos: %s", args.metric, average_score, len(scores))


if __name__ == "__main__":
    started = time.time()
    main()
    logging.info("Elapsed seconds: %.3f", time.time() - started)
