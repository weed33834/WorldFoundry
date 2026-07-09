"""Module for base_models -> diffusion_model -> video -> hunyuan_video -> utils -> helpers.py functionality."""

import collections.abc

from itertools import repeat


def _ntuple(n):
    """Helper function to ntuple.

    Args:
        n: The n.
    """
    def parse(x):
        """Parse.

        Args:
            x: The x.
        """
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


def as_tuple(x):
    """As tuple.

    Args:
        x: The x.
    """
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    if x is None or isinstance(x, (int, float, str)):
        return (x,)
    else:
        raise ValueError(f"Unknown type {type(x)}")


def as_list_of_2tuple(x):
    """As list of 2tuple.

    Args:
        x: The x.
    """
    x = as_tuple(x)
    if len(x) == 1:
        x = (x[0], x[0])
    assert len(x) % 2 == 0, f"Expect even length, got {len(x)}."
    lst = []
    for i in range(0, len(x), 2):
        lst.append((x[i], x[i + 1]))
    return lst
