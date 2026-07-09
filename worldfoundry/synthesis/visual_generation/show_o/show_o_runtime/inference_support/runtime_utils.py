from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
from torchvision import transforms


def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> list[tuple[str, Any]]:
    ret: list[tuple[str, Any]] = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> list[tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> list[tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for key, value in cfg.items_ex(resolve=resolve):
            if isinstance(value, DictConfig):
                ret.extend(handle_dict(key, value, resolve=resolve))
            elif isinstance(value, ListConfig):
                ret.extend(handle_list(key, value, resolve=resolve))
            else:
                ret.append((str(key), value))
    elif isinstance(cfg, ListConfig):
        for idx, value in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(value, DictConfig):
                ret.extend(handle_dict(idx, value, resolve=resolve))
            elif isinstance(value, ListConfig):
                ret.extend(handle_list(idx, value, resolve=resolve))
            else:
                ret.append((str(idx), value))
    else:
        raise TypeError(f"Expected OmegaConf DictConfig/ListConfig, got {type(cfg).__name__}")

    return ret


def image_transform(image, resolution: int = 256, normalize: bool = True):
    image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
    return image
