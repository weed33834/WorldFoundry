"""Module for base_models -> three_dimensions -> point_clouds -> pixelsplat_full -> src -> misc -> wandb_tools.py functionality."""

from pathlib import Path

import wandb


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_checkpoint(
    run_id: str,
    download_dir: Path,
    version: str | None,
) -> Path:
    """Download checkpoint.

    Args:
        run_id: The run id.
        download_dir: The download dir.
        version: The version.

    Returns:
        The return value.
    """
    api = wandb.Api()
    run = api.run(run_id)

    # Find the latest saved model checkpoint.
    chosen = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        # If no version is specified, use the latest.
        if version is None:
            if chosen is None or version_to_int(artifact) > version_to_int(chosen):
                chosen = artifact

        # If a specific verison is specified, look for it.
        elif version == artifact.version:
            chosen = artifact
            break

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
    chosen.download(root=root)
    return root / "model.ckpt"


def update_checkpoint_path(path: str | None, wandb_cfg: dict) -> Path | None:
    """Update checkpoint path.

    Args:
        path: The path.
        wandb_cfg: The wandb cfg.

    Returns:
        The return value.
    """
    if path is None:
        return None

    if not str(path).startswith("wandb://"):
        return Path(path)

    run_id, *version = path[len("wandb://") :].split(":")
    if len(version) == 0:
        version = None
    elif len(version) == 1:
        version = version[0]
    else:
        raise ValueError("Invalid version specifier!")

    project = wandb_cfg["project"]
    return download_checkpoint(
        f"{project}/{run_id}",
        Path("checkpoints"),
        version,
    )
