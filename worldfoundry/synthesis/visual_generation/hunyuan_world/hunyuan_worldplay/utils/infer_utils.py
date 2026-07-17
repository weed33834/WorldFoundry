from functools import wraps

import torch

from ..commons.infer_state import get_infer_state


def torch_compile_wrapper():
    """Compile a decorated method at most once, on its first compiled call.

    The previous implementation invoked :func:`torch.compile` for every model
    forward.  Apart from the Python overhead, that discarded Dynamo's callable
    wrapper and could repeatedly enter graph capture on the interactive hot
    path.  Keeping the compiled callable in the decorator closure lets Dynamo
    manage shape guards and graph variants as intended.
    """

    def decorator(func):
        compiled_func = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal compiled_func
            if get_infer_state() and get_infer_state().enable_torch_compile:
                if compiled_func is None:
                    compiled_func = torch.compile(func)
                return compiled_func(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator
