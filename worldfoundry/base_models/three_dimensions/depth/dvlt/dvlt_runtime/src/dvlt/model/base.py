# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module for base_models -> three_dimensions -> depth -> dvlt -> dvlt_runtime -> src -> dvlt -> model -> base.py functionality."""

import itertools
import os
import re
import tempfile
import urllib.parse
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import Tensor, nn

from dvlt.common.constants import DataField
from dvlt.util.download import download_file_from_url


logger = get_logger(__name__)


class Module(ABC):
    """Module implementation."""

    def __init__(self, *args, **kwargs):
        """Initialize an inference model wrapper."""
        super().__init__()
        self.model_file = "model.pt"
        self.model = self.build_model()

    @abstractmethod
    def build_model(self):
        """Build model."""
        raise NotImplementedError

    def setup_test(self, accelerator: Accelerator) -> None:
        """Setup test.

        Args:
            accelerator: The accelerator.

        Returns:
            The return value.
        """
        self.model = accelerator.unwrap_model(self.model)
        self.model.to(device=accelerator.device)
        self.model.eval()

    @torch.no_grad()
    def test_step(self, batch: dict, accelerator: Accelerator) -> dict:
        """Run prediction only. Callbacks handle preprocessing and evaluation."""
        assert len(batch[DataField.IMAGES]) == 1, "Only support BS=1 testing"
        predictions = self.predict(batch, accelerator)

        # Callbacks handle all preprocessing (indexing, scaling, alignment) and metrics
        return predictions

    @abstractmethod
    @torch.no_grad()
    def predict(self, batch: dict, accelerator: Accelerator) -> dict:
        """
        Run model inference and return standardized predictions.

        This method should perform model inference and return predictions using
        standardized keys from PredictionField. The outputs should be converted
        to generic formats (e.g., Camera objects) where applicable.

        Args:
            batch: Input batch containing images and other data

        Returns:
            Dictionary with standardized prediction keys and generic output formats
        """
        raise NotImplementedError

    def print_summary(self):
        """Print a summary of the model."""
        logger.info(self.model)

    def load_pretrained(
        self,
        pretrained_model_name_or_path: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        model_file: Optional[str] = None,
        strict: bool = False,
        remap: Optional[Dict[str, str]] = None,
        filter: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Load a model from a local file, local directory, URL, or HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: Either:
                - A direct path to a weights file (``.safetensors`` or ``.pt``)
                - A local directory containing ``model.safetensors`` / ``model.pt``
                - A URL to a model file
                - A model identifier from the HuggingFace Hub
            use_auth_token: Optional HuggingFace auth token
            revision: Optional git revision to use if loading from Hub
            model_file: Optional filename for the model weights (defaults to self.model_file)
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned by this module's state_dict() function.
            remap: Optional dictionary mapping regex patterns to replacement strings for state_dict key remapping
            filter: Optional list of regex patterns to filter out keys from state_dict
            **kwargs: Additional arguments passed to state_dict loading function
        """
        if model_file is not None:
            self.model_file = model_file

        # Parse the input to determine if it's a local file, directory, URL, or Hub model ID
        is_url = bool(urllib.parse.urlparse(pretrained_model_name_or_path).scheme)

        if os.path.isfile(pretrained_model_name_or_path):
            # Direct path to a weights file (e.g. .safetensors or .pt)
            model_path = pretrained_model_name_or_path
            logger.info(f"Loading model from local file: {model_path}")

        elif os.path.isdir(pretrained_model_name_or_path):
            # Local directory
            model_path = os.path.join(pretrained_model_name_or_path, self.model_file)
            # Check if the model was saved as .safetensors
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    pretrained_model_name_or_path, os.path.splitext(self.model_file)[0] + ".safetensors"
                )

            if not os.path.exists(model_path):
                raise ValueError(f"No model checkpoint found at {model_path}")
            logger.info(f"Loading model from local directory: {model_path}")

        elif is_url:
            try:
                # Create a temporary file to download to
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                    tmp_path = tmp_file.name

                # Download the file using the utility function
                logger.info(f"Downloading model from URL: {pretrained_model_name_or_path}")
                download_file_from_url(pretrained_model_name_or_path, tmp_path)
                model_path = tmp_path
            except Exception as e:
                raise ValueError(
                    f"Could not download model from URL {pretrained_model_name_or_path}. " f"Error: {str(e)}"
                ) from e

        else:
            try:
                # ``use_auth_token`` was renamed to ``token`` in huggingface_hub>=0.23
                # and removed entirely in more recent versions.
                model_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename=self.model_file,
                    token=use_auth_token,
                    revision=revision,
                )
                logger.info(f"Downloaded model from Hugging Face Hub: {pretrained_model_name_or_path}")
            except Exception as e:
                raise ValueError(
                    f"Could not load model from Hugging Face Hub {pretrained_model_name_or_path}. " f"Error: {str(e)}"
                ) from e

        # Load the model state dict - using safetensors if applicable
        if model_path.endswith(".safetensors"):
            with safe_open(model_path, framework="pt") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
        else:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True, **kwargs)

        # Handle different formats of saved state_dict
        if isinstance(state_dict, Dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # Handle Accelerate save_state format
        elif isinstance(state_dict, Dict) and "module" in state_dict:
            state_dict = state_dict["module"]

        # Remove module. prefix if it exists (happens when saved with DataParallel)
        if all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        # Apply key remapping if provided. Aggregate per-pattern counts so we
        # log a single summary line per pattern instead of one line per key.
        remap_counts: Dict[str, int] = {}
        if remap is not None:
            remapped_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                for old_pattern, new_pattern in remap.items():
                    new_key_candidate = re.sub(old_pattern, new_pattern, key)
                    if new_key_candidate != key:
                        new_key = new_key_candidate
                        remap_counts[old_pattern] = remap_counts.get(old_pattern, 0) + 1
                        break
                remapped_state_dict[new_key] = value
            state_dict = remapped_state_dict

        # Filter out keys matching filter_patterns; aggregate per-pattern counts.
        filter_counts: Dict[str, int] = {}
        if filter is not None:
            filtered_state_dict = {}
            for key, value in state_dict.items():
                should_filter = False
                for pattern in filter:
                    if re.search(pattern, key):
                        filter_counts[pattern] = filter_counts.get(pattern, 0) + 1
                        should_filter = True
                        break
                if not should_filter:
                    filtered_state_dict[key] = value
            state_dict = filtered_state_dict

        # Subclass hook: massage state_dict (e.g. rename/duplicate keys) before shape checks.
        state_dict = self._transform_state_dict(state_dict)

        # Drop any key whose tensor shape doesn't match the target module so load_state_dict doesn't raise.
        # These keys then flow through as missing, and downstream logic (e.g. reinit hooks) can handle them.
        model_sd = self.model.state_dict()
        shape_mismatched = [
            k for k, v in state_dict.items() if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)
        ]
        if shape_mismatched:
            state_dict = {k: v for k, v in state_dict.items() if k not in shape_mismatched}

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=strict)

        # Concise load summary: one line per remap/filter pattern, plus a
        # bottom-line tally of mismatched / missing / unexpected with a small
        # sample (full lists at DEBUG level).
        for pat, n in remap_counts.items():
            logger.info(f"Remapped {n} key(s) via pattern '{pat}' -> '{remap[pat]}'.")
        for pat, n in filter_counts.items():
            logger.info(f"Filtered {n} key(s) matching pattern '{pat}'.")
        if shape_mismatched:
            logger.warning(
                f"Shape mismatch on {len(shape_mismatched)} key(s); dropped before load. "
                f"Sample: {shape_mismatched[:3]}"
            )
            logger.debug(f"Full shape-mismatched key list: {shape_mismatched}")
        if missing_keys:
            logger.warning(f"Missing {len(missing_keys)} key(s) in checkpoint. Sample: {list(missing_keys)[:3]}")
            logger.debug(f"Full missing-keys list: {list(missing_keys)}")
        if unexpected_keys:
            logger.warning(
                f"Unexpected {len(unexpected_keys)} key(s) in checkpoint. Sample: {list(unexpected_keys)[:3]}"
            )
            logger.debug(f"Full unexpected-keys list: {list(unexpected_keys)}")
        logger.info(
            "Model loaded: "
            f"{len(state_dict)} loaded, "
            f"{len(missing_keys)} missing, "
            f"{len(unexpected_keys)} unexpected, "
            f"{len(shape_mismatched)} shape-mismatched."
        )

        # Clean up temporary file if we created one
        if is_url and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        return missing_keys, unexpected_keys

    def _transform_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Transform state dict before loading. Override in subclasses for custom transformations."""
        return state_dict


class Model(nn.Module):
    """Model implementation."""

    def __init__(self, *args, **kwargs):
        """Init."""
        super().__init__()

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        return get_parameter_device(self)

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


def get_parameter_device(parameter: torch.nn.Module) -> torch.device:
    """Get parameter device.

    Args:
        parameter: The parameter.

    Returns:
        The return value.
    """
    try:
        # If the onload device is not available due to no group offloading hooks, try to get the device
        # from the first parameter or buffer
        parameters_and_buffers = itertools.chain(parameter.parameters(), parameter.buffers())
        return next(parameters_and_buffers).device
    except StopIteration:
        # For torch.nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            """Find tensor attributes.

            Args:
                module: The module.

            Returns:
                The return value.
            """
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].device


def get_parameter_dtype(parameter: torch.nn.Module) -> torch.dtype:
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None

    for name, param in parameter.named_parameters():
        last_dtype = param.dtype
        if (
            hasattr(parameter, "_keep_in_fp32_modules")
            and parameter._keep_in_fp32_modules
            and any(m in name for m in parameter._keep_in_fp32_modules)
        ):
            continue

        if param.is_floating_point():
            return param.dtype

    for buffer in parameter.buffers():
        last_dtype = buffer.dtype
        if buffer.is_floating_point():
            return buffer.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype

    # For nn.DataParallel compatibility in PyTorch > 1.5
    def find_tensor_attributes(module: nn.Module) -> List[Tuple[str, Tensor]]:
        """Find tensor attributes.

        Args:
            module: The module.

        Returns:
            The return value.
        """
        tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
        return tuples

    gen = parameter._named_members(get_members_fn=find_tensor_attributes)
    last_tuple = None
    for tuple in gen:
        last_tuple = tuple
        if tuple[1].is_floating_point():
            return tuple[1].dtype

    if last_tuple is not None:
        # fallback to the last dtype
        return last_tuple[1].dtype
