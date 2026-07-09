from worldfoundry.base_models.diffusion_model.video.lvdm.utils import (
    get_obj_from_str,
    instantiate_from_config as _instantiate_from_config,
)


def import_class_from_string(class_path: str):
    return get_obj_from_str(class_path)


def instantiate_from_config(config) -> object:
    if "target" not in config:
        raise ValueError("Config must contain 'target' key specifying the class path.")
    return _instantiate_from_config(config)
