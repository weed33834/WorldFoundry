"""Module for base_models -> diffusion_model -> diffsynth -> pipelines -> svd_video.py functionality."""

from ..models import ModelManager, SVDImageEncoder, SVDUNet, SVDVAEEncoder, SVDVAEDecoder
from ..schedulers import ContinuousODEScheduler
from ..diffusion.base_pipeline import BasePipeline
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from einops import rearrange, repeat



class SVDVideoPipeline(BasePipeline):
    """Svd video pipeline implementation."""

    def __init__(self, device="cuda", torch_dtype=torch.float16):
        """Init.

        Args:
            device: The device.
            torch_dtype: The torch dtype.
        """
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = ContinuousODEScheduler()
        # models
        self.image_encoder: SVDImageEncoder = None
        self.unet: SVDUNet = None
        self.vae_encoder: SVDVAEEncoder = None
        self.vae_decoder: SVDVAEDecoder = None
    

    def fetch_models(self, model_manager: ModelManager):
        """Fetch models.

        Args:
            model_manager: The model manager.
        """
        self.image_encoder = model_manager.fetch_model("svd_image_encoder")
        self.unet = model_manager.fetch_model("svd_unet")
        self.vae_encoder = model_manager.fetch_model("svd_vae_encoder")
        self.vae_decoder = model_manager.fetch_model("svd_vae_decoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, **kwargs):
        """From model manager.

        Args:
            model_manager: The model manager.
        """
        pipe = SVDVideoPipeline(
            device=model_manager.device,
            torch_dtype=model_manager.torch_dtype
        )
        pipe.fetch_models(model_manager)
        return pipe
    

    def encode_image_with_clip(self, image):
        """Encode image with clip.

        Args:
            image: The image.
        """
        image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
        image = SVDCLIPImageProcessor().resize_with_antialiasing(image, (224, 224))
        image = (image + 1.0) / 2.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1, 3, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1, 3, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        image = (image - mean) / std
        image_emb = self.image_encoder(image)
        return image_emb
    

    def encode_image_with_vae(self, image, noise_aug_strength, seed=None):
        """Encode image with vae.

        Args:
            image: The image.
            noise_aug_strength: The noise aug strength.
            seed: The seed.
        """
        image = self.preprocess_image(image).to(device=self.device, dtype=self.torch_dtype)
        noise = self.generate_noise(image.shape, seed=seed, device=self.device, dtype=self.torch_dtype)
        image = image + noise_aug_strength * noise
        image_emb = self.vae_encoder(image) / self.vae_encoder.scaling_factor
        return image_emb
    

    def encode_video_with_vae(self, video):
        """Encode video with vae.

        Args:
            video: The video.
        """
        video = torch.concat([self.preprocess_image(frame) for frame in video], dim=0)
        video = rearrange(video, "T C H W -> 1 C T H W")
        video = video.to(device=self.device, dtype=self.torch_dtype)
        latents = self.vae_encoder.encode_video(video)
        latents = rearrange(latents[0], "C T H W -> T C H W")
        return latents
    

    def tensor2video(self, frames):
        """Tensor2video.

        Args:
            frames: The frames.
        """
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    

    def calculate_noise_pred(
        self,
        latents,
        timestep,
        add_time_id,
        cfg_scales,
        image_emb_vae_posi, image_emb_clip_posi,
        image_emb_vae_nega, image_emb_clip_nega
    ):
        """Calculate noise pred.

        Args:
            latents: The latents.
            timestep: The timestep.
            add_time_id: The add time id.
            cfg_scales: The cfg scales.
            image_emb_vae_posi: The image emb vae posi.
            image_emb_clip_posi: The image emb clip posi.
            image_emb_vae_nega: The image emb vae nega.
            image_emb_clip_nega: The image emb clip nega.
        """
        # Positive side
        noise_pred_posi = self.unet(
            torch.cat([latents, image_emb_vae_posi], dim=1),
            timestep, image_emb_clip_posi, add_time_id
        )
        # Negative side
        noise_pred_nega = self.unet(
            torch.cat([latents, image_emb_vae_nega], dim=1),
            timestep, image_emb_clip_nega, add_time_id
        )

        # Classifier-free guidance
        noise_pred = noise_pred_nega + cfg_scales * (noise_pred_posi - noise_pred_nega)

        return noise_pred
    

    def post_process_latents(self, latents, post_normalize=True, contrast_enhance_scale=1.0):
        """Post process latents.

        Args:
            latents: The latents.
            post_normalize: The post normalize.
            contrast_enhance_scale: The contrast enhance scale.
        """
        if post_normalize:
            mean, std = latents.mean(), latents.std()
            latents = (latents - latents.mean(dim=[1, 2, 3], keepdim=True)) / latents.std(dim=[1, 2, 3], keepdim=True) * std + mean
        latents = latents * contrast_enhance_scale
        return latents


    @torch.no_grad()
    def __call__(
        self,
        input_image=None,
        input_video=None,
        mask_frames=[],
        mask_frame_ids=[],
        min_cfg_scale=1.0,
        max_cfg_scale=3.0,
        denoising_strength=1.0,
        num_frames=25,
        height=576,
        width=1024,
        fps=7,
        motion_bucket_id=127,
        noise_aug_strength=0.02,
        num_inference_steps=20,
        post_normalize=True,
        contrast_enhance_scale=1.2,
        seed=None,
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
    ):
        """Call.

        Args:
            input_image: The input image.
            input_video: The input video.
            mask_frames: The mask frames.
            mask_frame_ids: The mask frame ids.
            min_cfg_scale: The min cfg scale.
            max_cfg_scale: The max cfg scale.
            denoising_strength: The denoising strength.
            num_frames: The num frames.
            height: The height.
            width: The width.
            fps: The fps.
            motion_bucket_id: The motion bucket id.
            noise_aug_strength: The noise aug strength.
            num_inference_steps: The num inference steps.
            post_normalize: The post normalize.
            contrast_enhance_scale: The contrast enhance scale.
            seed: The seed.
            progress_bar_cmd: The progress bar cmd.
            progress_bar_st: The progress bar st.
        """
        height, width = self.check_resize_height_width(height, width)
        
        # Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength)

        # Prepare latent tensors
        noise = self.generate_noise((num_frames, 4, height//8, width//8), seed=seed, device=self.device, dtype=self.torch_dtype)
        if denoising_strength == 1.0:
            latents = noise.clone()
        else:
            latents = self.encode_video_with_vae(input_video)
            latents = self.scheduler.add_noise(latents, noise, self.scheduler.timesteps[0])

        # Prepare mask frames
        if len(mask_frames) > 0:
            mask_latents = self.encode_video_with_vae(mask_frames)

        # Encode image
        image_emb_clip_posi = self.encode_image_with_clip(input_image)
        image_emb_clip_nega = torch.zeros_like(image_emb_clip_posi)
        image_emb_vae_posi = repeat(self.encode_image_with_vae(input_image, noise_aug_strength, seed=seed), "B C H W -> (B T) C H W", T=num_frames)
        image_emb_vae_nega = torch.zeros_like(image_emb_vae_posi)

        # Prepare classifier-free guidance
        cfg_scales = torch.linspace(min_cfg_scale, max_cfg_scale, num_frames)
        cfg_scales = cfg_scales.reshape(num_frames, 1, 1, 1).to(device=self.device, dtype=self.torch_dtype)
        
        # Prepare positional id
        add_time_id = torch.tensor([[fps-1, motion_bucket_id, noise_aug_strength]], device=self.device)

        # Denoise
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):

            # Mask frames
            for frame_id, mask_frame_id in enumerate(mask_frame_ids):
                latents[mask_frame_id] = self.scheduler.add_noise(mask_latents[frame_id], noise[mask_frame_id], timestep)

            # Fetch model output
            noise_pred = self.calculate_noise_pred(
                latents, timestep, add_time_id, cfg_scales,
                image_emb_vae_posi, image_emb_clip_posi, image_emb_vae_nega, image_emb_clip_nega
            )

            # Forward Euler
            latents = self.scheduler.step(noise_pred, timestep, latents)
            
            # Update progress bar
            if progress_bar_st is not None:
                progress_bar_st.progress(progress_id / len(self.scheduler.timesteps))

        # Decode image
        latents = self.post_process_latents(latents, post_normalize=post_normalize, contrast_enhance_scale=contrast_enhance_scale)
        video = self.vae_decoder.decode_video(latents, progress_bar=progress_bar_cmd)
        video = self.tensor2video(video)

        return video



class SVDCLIPImageProcessor:
    """Svdclip image processor implementation."""
    def __init__(self):
        """Init."""
        pass

    def resize_with_antialiasing(self, input, size, interpolation="bicubic", align_corners=True):
        """Resize with antialiasing.

        Args:
            input: The input.
            size: The size.
            interpolation: The interpolation.
            align_corners: The align corners.
        """
        h, w = input.shape[-2:]
        factors = (h / size[0], w / size[1])

        # First, we have to determine sigma
        # Taken from skimage: https://github.com/scikit-image/scikit-image/blob/v0.19.2/skimage/transform/_warps.py#L171
        sigmas = (
            max((factors[0] - 1.0) / 2.0, 0.001),
            max((factors[1] - 1.0) / 2.0, 0.001),
        )

        # Now kernel size. Good results are for 3 sigma, but that is kind of slow. Pillow uses 1 sigma
        # https://github.com/python-pillow/Pillow/blob/master/src/libImaging/Resample.c#L206
        # But they do it in the 2 passes, which gives better results. Let's try 2 sigmas for now
        ks = int(max(2.0 * 2 * sigmas[0], 3)), int(max(2.0 * 2 * sigmas[1], 3))

        # Make sure it is odd
        if (ks[0] % 2) == 0:
            ks = ks[0] + 1, ks[1]

        if (ks[1] % 2) == 0:
            ks = ks[0], ks[1] + 1

        input = self._gaussian_blur2d(input, ks, sigmas)

        output = torch.nn.functional.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)
        return output


    def _compute_padding(self, kernel_size):
        """Compute padding tuple."""
        # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
        # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
        if len(kernel_size) < 2:
            raise AssertionError(kernel_size)
        computed = [k - 1 for k in kernel_size]

        # for even kernels we need to do asymmetric padding :(
        out_padding = 2 * len(kernel_size) * [0]

        for i in range(len(kernel_size)):
            computed_tmp = computed[-(i + 1)]

            pad_front = computed_tmp // 2
            pad_rear = computed_tmp - pad_front

            out_padding[2 * i + 0] = pad_front
            out_padding[2 * i + 1] = pad_rear

        return out_padding


    def _filter2d(self, input, kernel):
        """Helper function to filter2d.

        Args:
            input: The input.
            kernel: The kernel.
        """
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel = kernel[:, None, ...].to(device=input.device, dtype=input.dtype)

        tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

        height, width = tmp_kernel.shape[-2:]

        padding_shape: list[int] = self._compute_padding([height, width])
        input = torch.nn.functional.pad(input, padding_shape, mode="reflect")

        # kernel and input tensor reshape to align element-wise or batch-wise params
        tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
        input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

        # convolve the tensor with the kernel.
        output = torch.nn.functional.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

        out = output.view(b, c, h, w)
        return out


    def _gaussian(self, window_size: int, sigma):
        """Helper function to gaussian.

        Args:
            window_size: The window size.
            sigma: The sigma.
        """
        if isinstance(sigma, float):
            sigma = torch.tensor([[sigma]])

        batch_size = sigma.shape[0]

        x = (torch.arange(window_size, device=sigma.device, dtype=sigma.dtype) - window_size // 2).expand(batch_size, -1)

        if window_size % 2 == 0:
            x = x + 0.5

        gauss = torch.exp(-x.pow(2.0) / (2 * sigma.pow(2.0)))

        return gauss / gauss.sum(-1, keepdim=True)


    def _gaussian_blur2d(self, input, kernel_size, sigma):
        """Helper function to gaussian blur2d.

        Args:
            input: The input.
            kernel_size: The kernel size.
            sigma: The sigma.
        """
        if isinstance(sigma, tuple):
            sigma = torch.tensor([sigma], dtype=input.dtype)
        else:
            sigma = sigma.to(dtype=input.dtype)

        ky, kx = int(kernel_size[0]), int(kernel_size[1])
        bs = sigma.shape[0]
        kernel_x = self._gaussian(kx, sigma[:, 1].view(bs, 1))
        kernel_y = self._gaussian(ky, sigma[:, 0].view(bs, 1))
        out_x = self._filter2d(input, kernel_x[..., None, :])
        out = self._filter2d(out_x, kernel_y[..., None])

        return out
