import os
from pathlib import Path

from omegaconf import DictConfig


def _history_config_root() -> Path:
    override = os.environ.get("WORLDFOUNDRY_MME_VLA_CONFIG_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    from worldfoundry.core.io.paths import package_root

    return (
        package_root()
        / "data"
        / "models"
        / "runtime"
        / "configs"
        / "vla_va_wam"
        / "mme_vla"
        / "models"
        / "config"
        / "robomme"
    )


def get_history_config(history_config: str | DictConfig):
    if history_config in ["None", "none"]:
        return None
    if isinstance(history_config, str):
        import omegaconf

        path = Path(os.path.expandvars(history_config)).expanduser()
        if not path.is_absolute() and not path.exists():
            path = _history_config_root() / path
        if not path.is_file():
            raise FileNotFoundError(f"MME-VLA history config not found: {path}")
        return omegaconf.OmegaConf.load(path)
    elif isinstance(history_config, DictConfig):
        return history_config
    elif history_config is None:
        return None
    else:
        raise ValueError(f"Invalid history config: {history_config}")
