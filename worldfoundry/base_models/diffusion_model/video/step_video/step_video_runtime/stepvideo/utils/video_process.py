"""Module for base_models -> diffusion_model -> video -> step_video -> step_video_runtime -> stepvideo -> utils -> video_process.py functionality."""

import numpy as np
import datetime
import torch
import os
import imageio


class VideoProcessor:
    """Video processor implementation."""
    def __init__(self, save_path: str='./results', name_suffix: str=''):
        """Init.

        Args:
            save_path: The save path.
            name_suffix: The name suffix.
        """
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.name_suffix = name_suffix
    
    def crop2standard540p(self, vid_array):
        """Crop2standard540p.

        Args:
            vid_array: The vid array.
        """
        _, height, width, _ = vid_array.shape
        height_center = height//2
        width_center = width//2
        if width_center>height_center:  ## horizon mode
            return vid_array[:, height_center-270:height_center+270, width_center-480:width_center+480]
        elif width_center<height_center: ## portrait mode
            return vid_array[:, height_center-480:height_center+480, width_center-270:width_center+270]
        else:
            return vid_array

    def save_imageio_video(self, video_array: np.array, output_filename: str, fps=25, codec='libx264'):
        """Save imageio video.

        Args:
            video_array: The video array.
            output_filename: The output filename.
            fps: The fps.
            codec: The codec.
        """
        
        ffmpeg_params = [
            "-vf", "atadenoise=0a=0.1:0b=0.1:1a=0.1:1b=0.1",  # denoise
        ]
   
        with imageio.get_writer(output_filename, fps=fps, codec=codec, ffmpeg_params=ffmpeg_params) as vid_writer:
            for img_array in video_array:
                vid_writer.append_data(img_array)   
        
    
    def postprocess_video(self, video_tensor, output_file_name='', output_type="mp4", crop2standard540p=True):
        """Postprocess video.

        Args:
            video_tensor: The video tensor.
            output_file_name: The output file name.
            output_type: The output type.
            crop2standard540p: The crop2standard540p.
        """
        if len(self.name_suffix) == 0:
            video_path = os.path.join(self.save_path, f"{output_file_name}-{str(datetime.datetime.now())}.{output_type}")
        else:
            video_path = os.path.join(self.save_path, f"{output_file_name}-{self.name_suffix}.{output_type}")
        
        video_tensor = (video_tensor.cpu().clamp(-1, 1)+1)*127.5
        video_tensor = torch.cat([t for t in video_tensor], dim=-2)
        video_array = video_tensor.clamp(0, 255).to(torch.uint8).numpy().transpose(0,2,3,1)
        
        if crop2standard540p:
            video_array = self.crop2standard540p(video_array)

        self.save_imageio_video(video_array, video_path)
        print(f"Saved the generated video in {video_path}")
