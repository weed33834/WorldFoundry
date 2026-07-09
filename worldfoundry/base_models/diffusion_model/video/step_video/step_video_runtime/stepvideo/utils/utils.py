"""Module for base_models -> diffusion_model -> video -> step_video -> step_video_runtime -> stepvideo -> utils -> utils.py functionality."""

import numpy as np
import random
import torch
from functools import wraps
import torch.utils._device


def setup_seed(seed):
    """Setup seed.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EmptyInitOnDevice(torch.overrides.TorchFunctionMode):
    """Empty init on device implementation."""
    def __init__(self, device=None):
        """Init.

        Args:
            device: The device.
        """
        self.device = device

    def __torch_function__(self, func, types, args=(), kwargs=None):
        """Torch function.

        Args:
            func: The func.
            types: The types.
            args: The args.
            kwargs: The kwargs.
        """
        kwargs = kwargs or {}
        if getattr(func, '__module__', None) == 'torch.nn.init':
            if 'tensor' in kwargs:
                return kwargs['tensor']
            else:
                return args[0]
        if self.device is not None and func in torch.utils._device._device_constructors() and kwargs.get('device') is None:
            kwargs['device'] = self.device
        return func(*args, **kwargs)
    

def with_empty_init(func):
    """With empty init.

    Args:
        func: The func.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrapper."""
        with EmptyInitOnDevice('cpu'):
            return func(*args, **kwargs)
    return wrapper


    
    
def culens2mask(
    cu_seqlens=None,
    cu_seqlens_kv=None,
    max_seqlen=None,
    max_seqlen_kv=None,
    is_causal=False
):
    """Culens2mask.

    Args:
        cu_seqlens: The cu seqlens.
        cu_seqlens_kv: The cu seqlens kv.
        max_seqlen: The max seqlen.
        max_seqlen_kv: The max seqlen kv.
        is_causal: The is causal.
    """
    assert len(cu_seqlens) == len(cu_seqlens_kv); "q k v should have same bsz..."
    bsz = len(cu_seqlens) - 1
    seqlens = cu_seqlens[1:]-cu_seqlens[:-1]
    seqlens_kv = cu_seqlens_kv[1:]-cu_seqlens_kv[:-1]
    
    attn_mask = torch.zeros(bsz, max_seqlen, max_seqlen_kv, dtype=torch.bool)
    for i, (seq_len, seq_len_kv) in enumerate(zip(seqlens, seqlens_kv)):
        if is_causal:
            attn_mask[i, :seq_len, :seq_len_kv] = torch.triu(torch.ones(seq_len, seq_len_kv), diagonal=1).bool()
        else:
            attn_mask[i, :seq_len, :seq_len_kv] = torch.ones([seq_len, seq_len_kv], dtype=torch.bool)

    return attn_mask
    
    
