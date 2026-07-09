"""Module for base_models -> diffusion_model -> diffsynth -> controlnets -> controlnet_unit.py functionality."""

import torch
import numpy as np
from .processors import Processor_id


class ControlNetConfigUnit:
    """Control net config unit implementation."""
    def __init__(self, processor_id: Processor_id, model_path, scale=1.0, skip_processor=False):
        """Init.

        Args:
            processor_id: The processor id.
            model_path: The model path.
            scale: The scale.
            skip_processor: The skip processor.
        """
        self.processor_id = processor_id
        self.model_path = model_path
        self.scale = scale
        self.skip_processor = skip_processor


class ControlNetUnit:
    """Control net unit implementation."""
    def __init__(self, processor, model, scale=1.0):
        """Init.

        Args:
            processor: The processor.
            model: The model.
            scale: The scale.
        """
        self.processor = processor
        self.model = model
        self.scale = scale


class MultiControlNetManager:
    """Multi control net manager implementation."""
    def __init__(self, controlnet_units=[]):
        """Init.

        Args:
            controlnet_units: The controlnet units.
        """
        self.processors = [unit.processor for unit in controlnet_units]
        self.models = [unit.model for unit in controlnet_units]
        self.scales = [unit.scale for unit in controlnet_units]

    def cpu(self):
        """Cpu."""
        for model in self.models:
            model.cpu()

    def to(self, device):
        """To.

        Args:
            device: The device.
        """
        for model in self.models:
            model.to(device)
        for processor in self.processors:
            processor.to(device)
    
    def process_image(self, image, processor_id=None):
        """Process image.

        Args:
            image: The image.
            processor_id: The processor id.
        """
        if processor_id is None:
            processed_image = [processor(image) for processor in self.processors]
        else:
            processed_image = [self.processors[processor_id](image)]
        processed_image = torch.concat([
            torch.Tensor(np.array(image_, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
            for image_ in processed_image
        ], dim=0)
        return processed_image
    
    def __call__(
        self,
        sample, timestep, encoder_hidden_states, conditionings,
        tiled=False, tile_size=64, tile_stride=32, **kwargs
    ):
        """Call.

        Args:
            sample: The sample.
            timestep: The timestep.
            encoder_hidden_states: The encoder hidden states.
            conditionings: The conditionings.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
        """
        res_stack = None
        for processor, conditioning, model, scale in zip(self.processors, conditionings, self.models, self.scales):
            res_stack_ = model(
                sample, timestep, encoder_hidden_states, conditioning, **kwargs,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
                processor_id=processor.processor_id
            )
            res_stack_ = [res * scale for res in res_stack_]
            if res_stack is None:
                res_stack = res_stack_
            else:
                res_stack = [i + j for i, j in zip(res_stack, res_stack_)]
        return res_stack


class FluxMultiControlNetManager(MultiControlNetManager):
    """Flux multi control net manager implementation."""
    def __init__(self, controlnet_units=[]):
        """Init.

        Args:
            controlnet_units: The controlnet units.
        """
        super().__init__(controlnet_units=controlnet_units)

    def process_image(self, image, processor_id=None):
        """Process image.

        Args:
            image: The image.
            processor_id: The processor id.
        """
        if processor_id is None:
            processed_image = [processor(image) for processor in self.processors]
        else:
            processed_image = [self.processors[processor_id](image)]
        return processed_image

    def __call__(self, conditionings, **kwargs):
        """Call.

        Args:
            conditionings: The conditionings.
        """
        res_stack, single_res_stack = None, None
        for processor, conditioning, model, scale in zip(self.processors, conditionings, self.models, self.scales):
            res_stack_, single_res_stack_ = model(controlnet_conditioning=conditioning, processor_id=processor.processor_id, **kwargs)
            res_stack_ = [res * scale for res in res_stack_]
            single_res_stack_ = [res * scale for res in single_res_stack_]
            if res_stack is None:
                res_stack = res_stack_
                single_res_stack = single_res_stack_
            else:
                res_stack = [i + j for i, j in zip(res_stack, res_stack_)]
                single_res_stack = [i + j for i, j in zip(single_res_stack, single_res_stack_)]
        return res_stack, single_res_stack
