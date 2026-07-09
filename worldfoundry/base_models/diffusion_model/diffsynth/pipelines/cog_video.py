"""Module for base_models -> diffusion_model -> diffsynth -> pipelines -> cog_video.py functionality."""

from ..models import ModelManager, FluxTextEncoder2, CogDiT, CogVAEEncoder, CogVAEDecoder
from ..prompters import CogPrompter
from ..schedulers import EnhancedDDIMScheduler
from ..diffusion.base_pipeline import BasePipeline
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange



class CogVideoPipeline(BasePipeline):
    """Cog video pipeline implementation."""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        """Init.

        Args:
            device: The device.
            torch_dtype: The torch dtype.
        """
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = EnhancedDDIMScheduler(rescale_zero_terminal_snr=True, prediction_type="v_prediction")
        self.prompter = CogPrompter()
        # models
        self.text_encoder: FluxTextEncoder2 = None
        self.dit: CogDiT = None
        self.vae_encoder: CogVAEEncoder = None
        self.vae_decoder: CogVAEDecoder = None
    

    def fetch_models(self, model_manager: ModelManager, prompt_refiner_classes=[]):
        """Fetch models.

        Args:
            model_manager: The model manager.
            prompt_refiner_classes: The prompt refiner classes.
        """
        self.text_encoder = model_manager.fetch_model("flux_text_encoder_2")
        self.dit = model_manager.fetch_model("cog_dit")
        self.vae_encoder = model_manager.fetch_model("cog_vae_encoder")
        self.vae_decoder = model_manager.fetch_model("cog_vae_decoder")
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.load_prompt_refiners(model_manager, prompt_refiner_classes)


    @staticmethod
    def from_model_manager(model_manager: ModelManager, prompt_refiner_classes=[]):
        """From model manager.

        Args:
            model_manager: The model manager.
            prompt_refiner_classes: The prompt refiner classes.
        """
        pipe = CogVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype
        )
        pipe.fetch_models(model_manager, prompt_refiner_classes)
        return pipe
    

    def tensor2video(self, frames):
        """Tensor2video.

        Args:
            frames: The frames.
        """
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    

    def encode_prompt(self, prompt, positive=True):
        """Encode prompt.

        Args:
            prompt: The prompt.
            positive: The positive.
        """
        prompt_emb = self.prompter.encode_prompt(prompt, device=self.device, positive=positive)
        return {"prompt_emb": prompt_emb}
    

    def prepare_extra_input(self, latents):
        """Prepare extra input.

        Args:
            latents: The latents.
        """
        return {"image_rotary_emb": self.dit.prepare_rotary_positional_embeddings(latents.shape[3], latents.shape[4], latents.shape[2], device=self.device)}


    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_video=None,
        cfg_scale=7.0,
        denoising_strength=1.0,
        num_frames=49,
        height=480,
        width=720,
        num_inference_steps=20,
        tiled=False,
        tile_size=(60, 90),
        tile_stride=(30, 45),
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        """Call.

        Args:
            prompt: The prompt.
            negative_prompt: The negative prompt.
            input_video: The input video.
            cfg_scale: The cfg scale.
            denoising_strength: The denoising strength.
            num_frames: The num frames.
            height: The height.
            width: The width.
            num_inference_steps: The num inference steps.
            tiled: The tiled.
            tile_size: The tile size.
            tile_stride: The tile stride.
            seed: The seed.
            progress_bar_cmd: The progress bar cmd.
            progress_bar_st: The progress bar st.
        """
        height, width = self.check_resize_height_width(height, width)
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)

        # Prepare latent tensors
        noise = self.generate_noise((1, 16, num_frames // 4 + 1, height//8, width//8), seed=seed, device="cpu", dtype=self.torch_dtype)
        
        if denoising_strength == 1.0:
            latents = noise.clone()
        else:
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.vae_encoder.encode_video(input_video, **tiler_kwargs, progress_bar=progress_bar_cmd).to(dtype=self.torch_dtype)
            latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])
        if not tiled: latents = latents.to(self.device)

        # Encode prompt
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)

        # Extra input
        extra_input = self.prepare_extra_input(latents)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(self.device)

            # Classifier-free guidance
            noise_pred_posi = self.dit(
                latents, timestep=timestep, **prompt_emb_posi, **tiler_kwargs, **extra_input
            )
            if cfg_scale != 1.0:
                noise_pred_nega = self.dit(
                    latents, timestep=timestep, **prompt_emb_nega, **tiler_kwargs, **extra_input
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # DDIM
            latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents)
            
            # Update progress bar
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        video = self.vae_decoder.decode_video(latents.to("cpu"), **tiler_kwargs, progress_bar=progress_bar_cmd)
        video = self.tensor2video(video[0])

        return video
