"""Small VRAM accounting and dynamic model swap helpers."""

from __future__ import annotations

import torch

cpu = torch.device("cpu")
gpu = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else cpu
gpu_complete_modules: list[torch.nn.Module] = []


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, **kwargs) -> None:
        original_class = module.__class__
        module.__dict__["forge_backup_original_class"] = original_class

        def hacked_get_attr(self, name: str):
            if "_parameters" in self.__dict__:
                parameters = self.__dict__["_parameters"]
                if name in parameters:
                    parameter = parameters[name]
                    if parameter is None:
                        return None
                    if parameter.__class__ == torch.nn.Parameter:
                        return torch.nn.Parameter(parameter.to(**kwargs), requires_grad=parameter.requires_grad)
                    return parameter.to(**kwargs)
            if "_buffers" in self.__dict__:
                buffers = self.__dict__["_buffers"]
                if name in buffers:
                    return buffers[name].to(**kwargs)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type(
            "DynamicSwap_" + original_class.__name__,
            (original_class,),
            {"__getattr__": hacked_get_attr},
        )

    @staticmethod
    def _uninstall_module(module: torch.nn.Module) -> None:
        if "forge_backup_original_class" in module.__dict__:
            module.__class__ = module.__dict__.pop("forge_backup_original_class")

    @staticmethod
    def install_model(model: torch.nn.Module, **kwargs) -> None:
        for module in model.modules():
            DynamicSwapInstaller._install_module(module, **kwargs)

    @staticmethod
    def uninstall_model(model: torch.nn.Module) -> None:
        for module in model.modules():
            DynamicSwapInstaller._uninstall_module(module)


def patched_diffusers_current_device(model: torch.nn.Module, target_device: torch.device) -> None:
    if hasattr(model, "scale_shift_table"):
        model.scale_shift_table.data = model.scale_shift_table.data.to(target_device)
        return

    for _, module in model.named_modules():
        if hasattr(module, "weight"):
            module.to(target_device)
            return


fake_diffusers_current_device = patched_diffusers_current_device


def get_cuda_free_memory_gb(device=None) -> float:
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = gpu

    memory_stats = torch.cuda.memory_stats(device)
    bytes_active = memory_stats["active_bytes.all.current"]
    bytes_reserved = memory_stats["reserved_bytes.all.current"]
    bytes_free_cuda, _ = torch.cuda.mem_get_info(device)
    bytes_inactive_reserved = bytes_reserved - bytes_active
    bytes_total_available = bytes_free_cuda + bytes_inactive_reserved
    return bytes_total_available / (1024**3)


def log_gpu_memory(stage: str, device=None, rank: int = 0) -> None:
    if not torch.cuda.is_available():
        print(f"[rank {rank}] [GPU Memory][{stage}] CUDA unavailable")
        return
    if device is None:
        device = gpu

    free_gb = get_cuda_free_memory_gb(device)
    total_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    used_gb = total_gb - free_gb
    print(
        f"[rank {rank}] [GPU Memory][{stage}] "
        f"Used: {used_gb:.2f} GB | Free: {free_gb:.2f} GB | Total: {total_gb:.2f} GB"
    )


def move_model_to_device_with_memory_preservation(
    model: torch.nn.Module,
    target_device,
    preserved_memory_gb: float = 0,
) -> None:
    print(f"Moving {model.__class__.__name__} to {target_device} with preserved memory: {preserved_memory_gb} GB")

    for module in model.modules():
        if get_cuda_free_memory_gb(target_device) <= preserved_memory_gb:
            torch.cuda.empty_cache()
            return

        if hasattr(module, "weight"):
            module.to(device=target_device)

    model.to(device=target_device)
    torch.cuda.empty_cache()


def offload_model_from_device_for_memory_preservation(
    model: torch.nn.Module,
    target_device,
    preserved_memory_gb: float = 0,
) -> None:
    print(f"Offloading {model.__class__.__name__} from {target_device} to preserve memory: {preserved_memory_gb} GB")

    for module in model.modules():
        if get_cuda_free_memory_gb(target_device) >= preserved_memory_gb:
            torch.cuda.empty_cache()
            return

        if hasattr(module, "weight"):
            module.to(device=cpu)

    model.to(device=cpu)
    torch.cuda.empty_cache()


def unload_complete_models(*models: torch.nn.Module) -> None:
    for model in gpu_complete_modules + list(models):
        model.to(device=cpu)
        print(f"Unloaded {model.__class__.__name__} as complete.")

    gpu_complete_modules.clear()
    torch.cuda.empty_cache()


def load_model_as_complete(model: torch.nn.Module, target_device, unload: bool = True) -> None:
    if unload:
        unload_complete_models()

    model.to(device=target_device)
    print(f"Loaded {model.__class__.__name__} to {target_device} as complete.")

    gpu_complete_modules.append(model)


__all__ = [
    "DynamicSwapInstaller",
    "cpu",
    "fake_diffusers_current_device",
    "get_cuda_free_memory_gb",
    "gpu",
    "gpu_complete_modules",
    "load_model_as_complete",
    "log_gpu_memory",
    "move_model_to_device_with_memory_preservation",
    "offload_model_from_device_for_memory_preservation",
    "patched_diffusers_current_device",
    "unload_complete_models",
]
