"""
This module serves as an entry point or an export mechanism for core components related
to GigaWorld policy synthesis.

It re-exports the `GigaWorldPolicySynthesis` class, making it directly accessible
when importing from this package. This allows for a cleaner import path, e.g.,
`from some_package import GigaWorldPolicySynthesis` instead of
`from some_package.giga_world_policy_synthesis import GigaWorldPolicySynthesis`.
"""

from .giga_world_policy_synthesis import GigaWorldPolicySynthesis

# Defines the public API of this module. When a user does `from package import *`,
# only the names listed in __all__ will be imported.
__all__ = ["GigaWorldPolicySynthesis"]