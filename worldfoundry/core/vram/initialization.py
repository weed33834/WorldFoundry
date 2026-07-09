import torch
from contextlib import contextmanager


@contextmanager
def init_weights_on_device(device=torch.device("meta"), include_buffers: bool = False):
    old_register_parameter = torch.nn.Module.register_parameter
    old_register_buffer = torch.nn.Module.register_buffer if include_buffers else None

    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)

    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper

    patched_constructors = {}
    if include_buffers:
        patched_constructors = {
            name: getattr(torch, name)
            for name in ("empty", "zeros", "ones", "full")
        }

    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
            for name in patched_constructors:
                setattr(torch, name, patch_tensor_constructor(getattr(torch, name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers and old_register_buffer is not None:
            torch.nn.Module.register_buffer = old_register_buffer
            for name, old_fn in patched_constructors.items():
                setattr(torch, name, old_fn)


def skip_model_initialization(device=torch.device("meta")):
    return init_weights_on_device(device=device)


__all__ = ["init_weights_on_device", "skip_model_initialization"]
