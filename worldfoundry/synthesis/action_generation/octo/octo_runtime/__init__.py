from __future__ import annotations

import os
import sys
import shutil
import types

# Octo Studio runs are CPU/JAX by default. Set this before importing modules that
# touch JAX so CUDA/cuDNN mismatches in the isolated action env do not win.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from . import octo as _octo


def install_distutils_compat_shims() -> None:
    """Provide the distutils.spawn symbol expected by TensorFlow on slim Python builds."""
    try:
        import distutils.spawn  # noqa: F401

        return
    except Exception:
        pass

    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = lambda executable, path=None: shutil.which(executable, path=path)
    sys.modules.setdefault("distutils.spawn", spawn_module)
    try:
        import distutils as _distutils

        setattr(_distutils, "spawn", spawn_module)
    except Exception:
        return


def install_jax_compat_shims() -> None:
    """Install small JAX/Flax compatibility aliases used by the Octo runtime."""
    try:
        import jax
        from jax._src import config as _jax_config

        if not hasattr(jax.config, "define_bool_state") and hasattr(_jax_config, "define_bool_state"):
            setattr(jax.config, "define_bool_state", _jax_config.define_bool_state)
        if not hasattr(jax.random, "KeyArray"):
            setattr(jax.random, "KeyArray", jax.Array)
    except Exception:
        return


def register_octo_alias() -> None:
    """Register the vendored Octo package under its upstream import name."""
    install_distutils_compat_shims()
    install_jax_compat_shims()
    sys.modules.setdefault("octo", _octo)


register_octo_alias()

__all__ = ["install_distutils_compat_shims", "install_jax_compat_shims", "register_octo_alias"]
