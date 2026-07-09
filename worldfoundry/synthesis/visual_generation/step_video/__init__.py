"""
This module serves as the `__init__.py` file for a package or subpackage.

Its primary purpose is to define the public API exposed when a user imports directly from this package.
It specifically re-exports the `StepVideoT2VSynthesis` class, making it directly accessible
from the package namespace. This simplifies imports for users, allowing them to write
`from package_name import StepVideoT2VSynthesis` instead of
`from package_name.step_video_t2v_synthesis import StepVideoT2VSynthesis`.
"""

from .step_video_t2v_synthesis import StepVideoT2VSynthesis

# Defines the public interface of this package/module.
# When a user imports `*` from this package (e.g., `from my_package import *`),
# only the names listed in `__all__` will be imported.
# In this case, it explicitly exposes `StepVideoT2VSynthesis`.
__all__ = ["StepVideoT2VSynthesis"]