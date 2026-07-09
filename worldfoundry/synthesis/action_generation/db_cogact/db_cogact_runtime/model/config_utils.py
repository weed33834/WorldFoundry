from __future__ import annotations

from typing import List


def require_config_keys(required_keys: List[str]):
    def decorator(func):
        def wrapper(config, *args, **kwargs):
            missing = [key for key in required_keys if not hasattr(config, key)]
            if missing:
                raise ValueError(f"Missing required config keys: {missing}")
            return func(config, *args, **kwargs)

        return wrapper

    return decorator
