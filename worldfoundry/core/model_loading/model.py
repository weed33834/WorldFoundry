"""High-level model construction with optional VRAM management and disk mapping."""

import torch
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils import ContextManagers

from worldfoundry.core.model_loading.file import load_state_dict
from worldfoundry.core.vram.disk_map import DiskMap
from worldfoundry.core.vram.initialization import skip_model_initialization
from worldfoundry.core.vram.layers import enable_vram_management


def load_model(
    model_class,
    path,
    config=None,
    torch_dtype=torch.bfloat16,
    device="cpu",
    state_dict_converter=None,
    use_disk_map=False,
    module_map=None,
    vram_config=None,
    vram_limit=None,
    state_dict=None,
):
    """Construct a model, assign checkpoint weights, and finalize inference placement.

    Args:
        model_class: PyTorch module class to instantiate.
        path: Checkpoint path consumed by ``load_state_dict`` or ``DiskMap``.
        config: Keyword arguments passed to ``model_class``.
        torch_dtype: Final model dtype.
        device: Final model device when fine-grained VRAM management is absent.
        state_dict_converter: Optional key/shape converter applied before assignment.
        use_disk_map: Read parameter tensors lazily instead of loading a full state dict.
        module_map: Source-class to VRAM-wrapper mapping. Enabling it delegates
            placement to ``enable_vram_management``.
        vram_config: Offload/onload/preparing/computation placement dictionary.
        vram_limit: Optional used-memory limit in GiB for wrappers.
        state_dict: Already-loaded weights; takes precedence over ``path``.

    Returns:
        An eval-mode model with weights assigned.

    Notes:
        Construction uses meta-device initialization where possible. DeepSpeed
        ZeRO-3 receives its specialized state-dict assignment path.
    """
    config = {} if config is None else config
    try:
        with ContextManagers(get_init_context(torch_dtype=torch_dtype, device=device)):
            model = model_class(**config)
    except NotImplementedError as exc:
        # Some third-party-compatible modules move their parameters inside
        # ``__init__``.  That is incompatible with our meta-parameter
        # registration hook because PyTorch cannot copy an unmaterialized
        # tensor to CPU/CUDA.  Retry only this precise incompatibility with a
        # normal allocation; all other construction errors must remain loud.
        if is_deepspeed_zero3_enabled() or "Cannot copy out of meta tensor" not in str(exc):
            raise
        model = model_class(**config)
    # What is `module_map`?
    # This is a module mapping table for VRAM management.
    if module_map is not None:
        devices = [
            vram_config["offload_device"],
            vram_config["onload_device"],
            vram_config["preparing_device"],
            vram_config["computation_device"],
        ]
        device = [d for d in devices if d != "disk"][0]
        dtypes = [
            vram_config["offload_dtype"],
            vram_config["onload_dtype"],
            vram_config["preparing_dtype"],
            vram_config["computation_dtype"],
        ]
        dtype = [d for d in dtypes if d != "disk"][0]
        if vram_config["offload_device"] != "disk":
            if state_dict is None:
                state_dict = DiskMap(path, device, torch_dtype=dtype)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            else:
                state_dict = {i: state_dict[i] for i in state_dict}
            model.load_state_dict(state_dict, assign=True)
            model = enable_vram_management(
                model, module_map, vram_config=vram_config, disk_map=None, vram_limit=vram_limit
            )
        else:
            disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
            model = enable_vram_management(
                model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=vram_limit
            )
    else:
        # Why do we use `DiskMap`?
        # Sometimes a model file contains multiple models,
        # and DiskMap can load only the parameters of a single model,
        # avoiding the need to load all parameters in the file.
        if state_dict is not None:
            pass
        elif use_disk_map:
            state_dict = DiskMap(path, device, torch_dtype=torch_dtype)
        else:
            state_dict = load_state_dict(path, torch_dtype, device)
        # Why do we use `state_dict_converter`?
        # Some models are saved in complex formats,
        # and we need to convert the state dict into the appropriate format.
        if state_dict_converter is not None:
            state_dict = state_dict_converter(state_dict)
        else:
            state_dict = {i: state_dict[i] for i in state_dict}
        # Why does DeepSpeed ZeRO Stage 3 need to be handled separately?
        # Because at this stage, model parameters are partitioned across multiple GPUs.
        # Loading them directly could lead to excessive GPU memory consumption.
        if is_deepspeed_zero3_enabled():
            from transformers.integrations.deepspeed import _load_state_dict_into_zero3_model

            _load_state_dict_into_zero3_model(model, state_dict)
        else:
            model.load_state_dict(state_dict, assign=True)
        # Why do we call `to()`?
        # Because some models override the behavior of `to()`,
        # especially those from libraries like Transformers.
        model = model.to(dtype=torch_dtype, device=device)
    if hasattr(model, "eval"):
        model = model.eval()
    return model


def load_model_with_disk_offload(
    model_class, path, config=None, torch_dtype=torch.bfloat16, device="cpu", state_dict_converter=None, module_map=None
):
    """Construct a model whose inactive weights remain disk-backed.

    Args:
        model_class: PyTorch module class to instantiate on the meta device.
        path: Checkpoint path or paths indexed by ``DiskMap``.
        config: Keyword arguments passed to ``model_class``.
        torch_dtype: Computation dtype.
        device: Preparing and computation device.
        state_dict_converter: Optional checkpoint-key converter.
        module_map: Required source-class to wrapper-class mapping.

    Returns:
        Eval-mode model with disk-backed VRAM wrappers installed.
    """
    if isinstance(path, str):
        path = [path]
    config = {} if config is None else config
    with skip_model_initialization():
        model = model_class(**config)
    if hasattr(model, "eval"):
        model = model.eval()
    disk_map = DiskMap(path, device, state_dict_converter=state_dict_converter)
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": "disk",
        "onload_device": "disk",
        "preparing_dtype": getattr(torch, "float8_e4m3fn", torch.bfloat16),
        "preparing_device": device,
        "computation_dtype": torch_dtype,
        "computation_device": device,
    }
    enable_vram_management(model, module_map, vram_config=vram_config, disk_map=disk_map, vram_limit=80)
    return model


def get_init_context(torch_dtype, device):
    if is_deepspeed_zero3_enabled():
        import deepspeed
        from transformers.modeling_utils import set_zero3_state

        # Why do we use "deepspeed.zero.Init"?
        # Weight segmentation of the model can be performed on the CPU side
        # and loading the segmented weights onto the computing card
        init_contexts = [deepspeed.zero.Init(remote_device=device, dtype=torch_dtype), set_zero3_state()]
    else:
        # Why do we use `skip_model_initialization`?
        # It skips the random initialization of model parameters,
        # thereby speeding up model loading and avoiding excessive memory usage.
        init_contexts = [skip_model_initialization()]

    return init_contexts


__all__ = ["get_init_context", "load_model", "load_model_with_disk_offload"]
