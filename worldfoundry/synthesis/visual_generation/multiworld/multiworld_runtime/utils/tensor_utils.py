import torch


def auto_match_dim(source, target):
    # source: 1D tensor, numel() == one of the dim in target 
    if source.numel() == 1: 
        return source
    for dim_idx in range(len(target.shape)):
        if source.numel() == target.shape[dim_idx]:
            shape = [1] * len(target.shape)
            shape[dim_idx] = source.numel()
            return source.view(shape)
    raise ValueError(f"Cannot match dim. with source shape: {source.shape}, target shape: {target.shape}. To use this function, must satisfy that source.numel() == one of the dim in target.")


def move_action_to_device(obj, device: str | torch.device):
    """Move tensors in nested action dictionaries to the inference device."""
    if isinstance(obj, torch.Tensor):
        target_dtype = torch.bfloat16 if obj.is_floating_point() else torch.long
        return obj.to(device, dtype=target_dtype)

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = move_action_to_device(value, device)
        return obj

    if isinstance(obj, int):
        return [obj]

    return obj
