from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from worldfoundry.core import jsonable
from worldfoundry.core.io.paths import resolve_data_path


DEFAULT_CONFIG_DIR = resolve_data_path("models", "runtime", "configs", "le_wm", "config", "eval")


def _load_official_modules():
    import stable_pretraining as spt
    import stable_worldmodel as swm

    return spt, swm


def _image_transform(cfg: DictConfig, spt: Any):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def _episodes_length(dataset, episodes: np.ndarray) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def _hdf5_dataset_class(swm: Any):
    cls = getattr(swm.data, "HDF5Dataset", None)
    if cls is not None:
        return cls
    try:
        from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    except Exception as exc:  # pragma: no cover - only hit on broken runtime installs
        raise RuntimeError(
            "stable_worldmodel no longer exports swm.data.HDF5Dataset and its "
            "internal HDF5 reader could not be imported. Install a stable-worldmodel "
            "build with HDF5 format support."
        ) from exc
    return HDF5Dataset


def _dataset_candidates(dataset_name: str, cache_dir: Path) -> list[Path]:
    raw = Path(dataset_name).expanduser()
    if raw.suffix in {".h5", ".hdf5"} or raw.is_absolute():
        return [raw]
    return [
        cache_dir / f"{dataset_name}.h5",
        cache_dir / "datasets" / f"{dataset_name}.h5",
        cache_dir / dataset_name,
    ]


def _dataset(cfg: DictConfig, swm: Any):
    cache_dir = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()).expanduser()
    dataset_name = str(cfg.eval.dataset_name)
    cls = _hdf5_dataset_class(swm)
    candidates = _dataset_candidates(dataset_name, cache_dir)
    for candidate in candidates:
        if candidate.exists():
            dataset = cls(path=candidate, keys_to_cache=cfg.dataset.keys_to_cache)
            print(
                json.dumps(
                    {
                        "event": "leworldmodel_dataset_loaded",
                        "dataset_name": dataset_name,
                        "dataset_path": str(candidate),
                        "num_episodes": int(len(dataset.lengths)),
                        "num_rows": int(np.asarray(dataset.lengths).sum()),
                        "columns": list(dataset.column_names),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return dataset
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"LeWorldModel dataset {dataset_name!r} was not found. Checked: {checked}"
    )


def _overrides(args: argparse.Namespace) -> list[str]:
    overrides = [f"policy={args.policy}"]
    if args.cache_dir:
        overrides.append(f"cache_dir={args.cache_dir}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    if args.dataset_name:
        overrides.append(f"eval.dataset_name={args.dataset_name}")
    if args.num_eval is not None:
        overrides.append(f"eval.num_eval={args.num_eval}")
    if args.eval_budget is not None:
        overrides.append(f"eval.eval_budget={args.eval_budget}")
    if args.goal_offset_steps is not None:
        overrides.append(f"eval.goal_offset_steps={args.goal_offset_steps}")
    if args.img_size is not None:
        overrides.append(f"eval.img_size={args.img_size}")
    return overrides


def _load_config(args: argparse.Namespace) -> DictConfig:
    config_dir = Path(args.config_dir).expanduser().resolve()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name, overrides=_overrides(args))
    OmegaConf.set_struct(cfg, False)
    return cfg


def run(args: argparse.Namespace) -> dict[str, Any]:
    spt, swm = _load_official_modules()
    cfg = _load_config(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget, (
        "Planning horizon must be smaller than or equal to eval_budget"
    )

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    if "solver" in cfg and "device" in cfg.solver:
        cfg.solver.device = args.device

    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        "pixels": _image_transform(cfg, spt),
        "goal": _image_transform(cfg, spt),
    }

    dataset = _dataset(cfg, swm)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    if cfg.policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to(args.device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
        )
    else:
        policy = swm.policy.RandomPolicy()

    episode_len = _episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_by_episode = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_by_episode[ep_id] for ep_id in dataset.get_col_data(col_name)])
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    generator = np.random.default_rng(cfg.seed)
    random_episode_indices = generator.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[random_episode_indices])
    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    world.set_policy(policy)
    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=output_dir,
    )
    elapsed = time.time() - start_time

    text_path = output_dir / cfg.output.filename
    with text_path.open("a", encoding="utf-8") as handle:
        handle.write("\n==== CONFIG ====\n")
        handle.write(OmegaConf.to_yaml(cfg))
        handle.write("\n==== RESULTS ====\n")
        handle.write(f"metrics: {metrics}\n")
        handle.write(f"evaluation_time: {elapsed} seconds\n")

    payload = {
        "status": "succeeded",
        "config_name": args.config_name,
        "policy": cfg.policy,
        "dataset_name": cfg.eval.dataset_name,
        "metrics": jsonable(metrics),
        "evaluation_time": elapsed,
        "results_file": str(text_path),
        "output_dir": str(output_dir),
    }
    artifact_path = Path(args.artifact_path).expanduser().resolve() if args.artifact_path else output_dir / "worldfoundry_leworldmodel_result.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WorldFoundry LeWorldModel infer/eval wrapper")
    parser.add_argument("--config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--policy", default="random")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact-path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--num-eval", type=int, default=None)
    parser.add_argument("--eval-budget", type=int, default=None)
    parser.add_argument("--goal-offset-steps", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    payload = run(parse_args())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
