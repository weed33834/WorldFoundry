"""Module for base_models -> diffusion_model -> diffsynth -> processors -> RIFE.py functionality."""

import torch
import numpy as np
from PIL import Image
from .base import VideoProcessor


class RIFESmoother(VideoProcessor):
    """Rife smoother implementation."""
    def __init__(self, model, device="cuda", scale=1.0, batch_size=4, interpolate=True):
        """Init.

        Args:
            model: The model.
            device: The device.
            scale: The scale.
            batch_size: The batch size.
            interpolate: The interpolate.
        """
        self.model = model
        self.device = device

        # IFNet only does not support float16
        self.torch_dtype = torch.float32

        # Other parameters
        self.scale = scale
        self.batch_size = batch_size
        self.interpolate = interpolate

    @staticmethod
    def from_model_manager(model_manager, **kwargs):
        """From model manager.

        Args:
            model_manager: The model manager.
        """
        return RIFESmoother(model_manager.RIFE, device=model_manager.device, **kwargs)
    
    def process_image(self, image):
        """Process image.

        Args:
            image: The image.
        """
        width, height = image.size
        if width % 32 != 0 or height % 32 != 0:
            width = (width + 31) // 32
            height = (height + 31) // 32
            image = image.resize((width, height))
        image = torch.Tensor(np.array(image, dtype=np.float32)[:, :, [2,1,0]] / 255).permute(2, 0, 1)
        return image
    
    def process_images(self, images):
        """Process images.

        Args:
            images: The images.
        """
        images = [self.process_image(image) for image in images]
        images = torch.stack(images)
        return images
    
    def decode_images(self, images):
        """Decode images.

        Args:
            images: The images.
        """
        images = (images[:, [2,1,0]].permute(0, 2, 3, 1) * 255).clip(0, 255).numpy().astype(np.uint8)
        images = [Image.fromarray(image) for image in images]
        return images
    
    def process_tensors(self, input_tensor, scale=1.0, batch_size=4):
        """Process tensors.

        Args:
            input_tensor: The input tensor.
            scale: The scale.
            batch_size: The batch size.
        """
        output_tensor = []
        for batch_id in range(0, input_tensor.shape[0], batch_size):
            batch_id_ = min(batch_id + batch_size, input_tensor.shape[0])
            batch_input_tensor = input_tensor[batch_id: batch_id_]
            batch_input_tensor = batch_input_tensor.to(device=self.device, dtype=self.torch_dtype)
            flow, mask, merged = self.model(batch_input_tensor, [4/scale, 2/scale, 1/scale])
            output_tensor.append(merged[2].cpu())
        output_tensor = torch.concat(output_tensor, dim=0)
        return output_tensor

    @torch.no_grad()
    def __call__(self, rendered_frames, **kwargs):
        """Call.

        Args:
            rendered_frames: The rendered frames.
        """
        # Preprocess
        processed_images = self.process_images(rendered_frames)

        # Input
        input_tensor = torch.cat((processed_images[:-2], processed_images[2:]), dim=1)

        # Interpolate
        output_tensor = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
        
        if self.interpolate:
            # Blend
            input_tensor = torch.cat((processed_images[1:-1], output_tensor), dim=1)
            output_tensor = self.process_tensors(input_tensor, scale=self.scale, batch_size=self.batch_size)
            processed_images[1:-1] = output_tensor
        else:
            processed_images[1:-1] = (processed_images[1:-1] + output_tensor) / 2

        # To images
        output_images = self.decode_images(processed_images)
        if output_images[0].size != rendered_frames[0].size:
            output_images = [image.resize(rendered_frames[0].size) for image in output_images]
        return output_images
