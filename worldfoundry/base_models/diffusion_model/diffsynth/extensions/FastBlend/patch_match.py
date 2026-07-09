"""Module for base_models -> diffusion_model -> diffsynth -> extensions -> FastBlend -> patch_match.py functionality."""

from .cupy_kernels import remapping_kernel, patch_error_kernel, pairwise_patch_error_kernel
import numpy as np
import cupy as cp
import cv2
import torch
import torch.nn.functional as F

class PatchMatcher:
    """Patch matcher implementation."""
    def __init__(
        self, height, width, channel, minimum_patch_size,
        threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0,
        random_search_steps=3, random_search_range=4,
        use_mean_target_style=False, use_pairwise_patch_error=False,
        tracking_window_size=0
    ):
        """Init.

        Args:
            height: The height.
            width: The width.
            channel: The channel.
            minimum_patch_size: The minimum patch size.
            threads_per_block: The threads per block.
            num_iter: The num iter.
            gpu_id: The gpu id.
            guide_weight: The guide weight.
            random_search_steps: The random search steps.
            random_search_range: The random search range.
            use_mean_target_style: The use mean target style.
            use_pairwise_patch_error: The use pairwise patch error.
            tracking_window_size: The tracking window size.
        """
        self.height = height
        self.width = width
        self.channel = channel
        self.minimum_patch_size = minimum_patch_size
        self.threads_per_block = threads_per_block
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.guide_weight = guide_weight
        self.random_search_steps = random_search_steps
        self.random_search_range = random_search_range
        self.use_mean_target_style = use_mean_target_style
        self.use_pairwise_patch_error = use_pairwise_patch_error
        self.tracking_window_size = tracking_window_size

        self.patch_size_list = [minimum_patch_size + i*2 for i in range(num_iter)][::-1]
        self.pad_size = self.patch_size_list[0] // 2
        self.grid = (
            (height + threads_per_block - 1) // threads_per_block,
            (width + threads_per_block - 1) // threads_per_block
        )
        self.block = (threads_per_block, threads_per_block)

    def pad_image(self, image):
        """Pad image.

        Args:
            image: The image.
        """
        return cp.pad(image, ((0, 0), (self.pad_size, self.pad_size), (self.pad_size, self.pad_size), (0, 0)))

    def unpad_image(self, image):
        """Unpad image.

        Args:
            image: The image.
        """
        return image[:, self.pad_size: -self.pad_size, self.pad_size: -self.pad_size, :]

    def apply_nnf_to_image(self, nnf, source):
        """Apply nnf to image.

        Args:
            nnf: The nnf.
            source: The source.
        """
        batch_size = source.shape[0]
        target = cp.zeros((batch_size, self.height + self.pad_size * 2, self.width + self.pad_size * 2, self.channel), dtype=cp.float32)
        remapping_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source, nnf, target)
        )
        return target

    def get_patch_error(self, source, nnf, target):
        """Get patch error.

        Args:
            source: The source.
            nnf: The nnf.
            target: The target.
        """
        batch_size = source.shape[0]
        error = cp.zeros((batch_size, self.height, self.width), dtype=cp.float32)
        patch_error_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source, nnf, target, error)
        )
        return error

    def get_pairwise_patch_error(self, source, nnf):
        """Get pairwise patch error.

        Args:
            source: The source.
            nnf: The nnf.
        """
        batch_size = source.shape[0]//2
        error = cp.zeros((batch_size, self.height, self.width), dtype=cp.float32)
        source_a, nnf_a = source[0::2].copy(), nnf[0::2].copy()
        source_b, nnf_b = source[1::2].copy(), nnf[1::2].copy()
        pairwise_patch_error_kernel(
            self.grid + (batch_size,),
            self.block,
            (self.height, self.width, self.channel, self.patch_size, self.pad_size, source_a, nnf_a, source_b, nnf_b, error)
        )
        error = error.repeat(2, axis=0)
        return error

    def get_error(self, source_guide, target_guide, source_style, target_style, nnf):
        """Get error.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            target_style: The target style.
            nnf: The nnf.
        """
        error_guide = self.get_patch_error(source_guide, nnf, target_guide)
        if self.use_mean_target_style:
            target_style = self.apply_nnf_to_image(nnf, source_style)
            target_style = target_style.mean(axis=0, keepdims=True)
            target_style = target_style.repeat(source_guide.shape[0], axis=0)
        if self.use_pairwise_patch_error:
            error_style = self.get_pairwise_patch_error(source_style, nnf)
        else:
            error_style = self.get_patch_error(source_style, nnf, target_style)
        error = error_guide * self.guide_weight + error_style
        return error

    def clamp_bound(self, nnf):
        """Clamp bound.

        Args:
            nnf: The nnf.
        """
        nnf[:,:,:,0] = cp.clip(nnf[:,:,:,0], 0, self.height-1)
        nnf[:,:,:,1] = cp.clip(nnf[:,:,:,1], 0, self.width-1)
        return nnf

    def random_step(self, nnf, r):
        """Random step.

        Args:
            nnf: The nnf.
            r: The r.
        """
        batch_size = nnf.shape[0]
        step = cp.random.randint(-r, r+1, size=(batch_size, self.height, self.width, 2), dtype=cp.int32)
        upd_nnf = self.clamp_bound(nnf + step)
        return upd_nnf

    def neighboor_step(self, nnf, d):
        """Neighboor step.

        Args:
            nnf: The nnf.
            d: The d.
        """
        if d==0:
            upd_nnf = cp.concatenate([nnf[:, :1, :], nnf[:, :-1, :]], axis=1)
            upd_nnf[:, :, :, 0] += 1
        elif d==1:
            upd_nnf = cp.concatenate([nnf[:, :, :1], nnf[:, :, :-1]], axis=2)
            upd_nnf[:, :, :, 1] += 1
        elif d==2:
            upd_nnf = cp.concatenate([nnf[:, 1:, :], nnf[:, -1:, :]], axis=1)
            upd_nnf[:, :, :, 0] -= 1
        elif d==3:
            upd_nnf = cp.concatenate([nnf[:, :, 1:], nnf[:, :, -1:]], axis=2)
            upd_nnf[:, :, :, 1] -= 1
        upd_nnf = self.clamp_bound(upd_nnf)
        return upd_nnf
        
    def shift_nnf(self, nnf, d):
        """Shift nnf.

        Args:
            nnf: The nnf.
            d: The d.
        """
        if d>0:
            d = min(nnf.shape[0], d)
            upd_nnf = cp.concatenate([nnf[d:]] + [nnf[-1:]] * d, axis=0)
        else:
            d = max(-nnf.shape[0], d)
            upd_nnf = cp.concatenate([nnf[:1]] * (-d) + [nnf[:d]], axis=0)
        return upd_nnf
    
    def track_step(self, nnf, d):
        """Track step.

        Args:
            nnf: The nnf.
            d: The d.
        """
        if self.use_pairwise_patch_error:
            upd_nnf = cp.zeros_like(nnf)
            upd_nnf[0::2] = self.shift_nnf(nnf[0::2], d)
            upd_nnf[1::2] = self.shift_nnf(nnf[1::2], d)
        else:
            upd_nnf = self.shift_nnf(nnf, d)
        return upd_nnf

    def C(self, n, m):
        """C.

        Args:
            n: The n.
            m: The m.
        """
        # not used
        c = 1
        for i in range(1, n+1):
            c *= i
        for i in range(1, m+1):
            c //= i
        for i in range(1, n-m+1):
            c //= i
        return c

    def bezier_step(self, nnf, r):
        """Bezier step.

        Args:
            nnf: The nnf.
            r: The r.
        """
        # not used
        n = r * 2 - 1
        upd_nnf = cp.zeros(shape=nnf.shape, dtype=cp.float32)
        for i, d in enumerate(list(range(-r, 0)) + list(range(1, r+1))):
            if d>0:
                ctl_nnf = cp.concatenate([nnf[d:]] + [nnf[-1:]] * d, axis=0)
            elif d<0:
                ctl_nnf = cp.concatenate([nnf[:1]] * (-d) + [nnf[:d]], axis=0)
            upd_nnf += ctl_nnf * (self.C(n, i) / 2**n)
        upd_nnf = self.clamp_bound(upd_nnf).astype(nnf.dtype)
        return upd_nnf

    def update(self, source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf):
        """Update.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            target_style: The target style.
            nnf: The nnf.
            err: The err.
            upd_nnf: The upd nnf.
        """
        upd_err = self.get_error(source_guide, target_guide, source_style, target_style, upd_nnf)
        upd_idx = (upd_err < err)
        nnf[upd_idx] = upd_nnf[upd_idx]
        err[upd_idx] = upd_err[upd_idx]
        return nnf, err

    def propagation(self, source_guide, target_guide, source_style, target_style, nnf, err):
        """Propagation.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            target_style: The target style.
            nnf: The nnf.
            err: The err.
        """
        for d in cp.random.permutation(4):
            upd_nnf = self.neighboor_step(nnf, d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err
        
    def random_search(self, source_guide, target_guide, source_style, target_style, nnf, err):
        """Random search.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            target_style: The target style.
            nnf: The nnf.
            err: The err.
        """
        for i in range(self.random_search_steps):
            upd_nnf = self.random_step(nnf, self.random_search_range)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err

    def track(self, source_guide, target_guide, source_style, target_style, nnf, err):
        """Track.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            target_style: The target style.
            nnf: The nnf.
            err: The err.
        """
        for d in range(1, self.tracking_window_size + 1):
            upd_nnf = self.track_step(nnf, d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
            upd_nnf = self.track_step(nnf, -d)
            nnf, err = self.update(source_guide, target_guide, source_style, target_style, nnf, err, upd_nnf)
        return nnf, err

    def iteration(self, source_guide, target_guide, source_style, target_style, nnf, err):
        """Iteration.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            target_style: The target style.
            nnf: The nnf.
            err: The err.
        """
        nnf, err = self.propagation(source_guide, target_guide, source_style, target_style, nnf, err)
        nnf, err = self.random_search(source_guide, target_guide, source_style, target_style, nnf, err)
        nnf, err = self.track(source_guide, target_guide, source_style, target_style, nnf, err)
        return nnf, err

    def estimate_nnf(self, source_guide, target_guide, source_style, nnf):
        """Estimate nnf.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
            nnf: The nnf.
        """
        with cp.cuda.Device(self.gpu_id):
            source_guide = self.pad_image(source_guide)
            target_guide = self.pad_image(target_guide)
            source_style = self.pad_image(source_style)
            for it in range(self.num_iter):
                self.patch_size = self.patch_size_list[it]
                target_style = self.apply_nnf_to_image(nnf, source_style)
                err = self.get_error(source_guide, target_guide, source_style, target_style, nnf)
                nnf, err = self.iteration(source_guide, target_guide, source_style, target_style, nnf, err)
            target_style = self.unpad_image(self.apply_nnf_to_image(nnf, source_style))
        return nnf, target_style


class PyramidPatchMatcher:
    """Pyramid patch matcher implementation."""
    def __init__(
        self, image_height, image_width, channel, minimum_patch_size,
        threads_per_block=8, num_iter=5, gpu_id=0, guide_weight=10.0,
        use_mean_target_style=False, use_pairwise_patch_error=False,
        tracking_window_size=0,
        initialize="identity"
    ):
        """Init.

        Args:
            image_height: The image height.
            image_width: The image width.
            channel: The channel.
            minimum_patch_size: The minimum patch size.
            threads_per_block: The threads per block.
            num_iter: The num iter.
            gpu_id: The gpu id.
            guide_weight: The guide weight.
            use_mean_target_style: The use mean target style.
            use_pairwise_patch_error: The use pairwise patch error.
            tracking_window_size: The tracking window size.
            initialize: The initialize.
        """
        maximum_patch_size = minimum_patch_size + (num_iter - 1) * 2
        self.pyramid_level = int(np.log2(min(image_height, image_width) / maximum_patch_size))
        self.pyramid_heights = []
        self.pyramid_widths = []
        self.patch_matchers = []
        self.minimum_patch_size = minimum_patch_size
        self.num_iter = num_iter
        self.gpu_id = gpu_id
        self.initialize = initialize
        for level in range(self.pyramid_level):
            height = image_height//(2**(self.pyramid_level - 1 - level))
            width = image_width//(2**(self.pyramid_level - 1 - level))
            self.pyramid_heights.append(height)
            self.pyramid_widths.append(width)
            self.patch_matchers.append(PatchMatcher(
                height, width, channel, minimum_patch_size=minimum_patch_size,
                threads_per_block=threads_per_block, num_iter=num_iter, gpu_id=gpu_id, guide_weight=guide_weight,
                use_mean_target_style=use_mean_target_style, use_pairwise_patch_error=use_pairwise_patch_error,
                tracking_window_size=tracking_window_size
            ))

    def resample_image(self, images, level):
        """Resample image.

        Args:
            images: The images.
            level: The level.
        """
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        images_torch = torch.as_tensor(images, device='cuda', dtype=torch.float32)
        images_torch = images_torch.permute(0, 3, 1, 2)
        images_resample = F.interpolate(images_torch, size=(height, width), mode='area', align_corners=None)
        images_resample = images_resample.permute(0, 2, 3, 1).contiguous()
        return cp.asarray(images_resample)

    def initialize_nnf(self, batch_size):
        """Initialize nnf.

        Args:
            batch_size: The batch size.
        """
        if self.initialize == "random":
            height, width = self.pyramid_heights[0], self.pyramid_widths[0]
            nnf = cp.stack([
                cp.random.randint(0, height, (batch_size, height, width), dtype=cp.int32),
                cp.random.randint(0, width, (batch_size, height, width), dtype=cp.int32)
            ], axis=3)
        elif self.initialize == "identity":
            height, width = self.pyramid_heights[0], self.pyramid_widths[0]
            nnf = cp.stack([
                cp.repeat(cp.arange(height), width).reshape(height, width),
                cp.tile(cp.arange(width), height).reshape(height, width)
            ], axis=2)
            nnf = cp.stack([nnf] * batch_size)
        else:
            raise NotImplementedError()
        return nnf

    def update_nnf(self, nnf, level):
        """Update nnf.

        Args:
            nnf: The nnf.
            level: The level.
        """
        # upscale
        nnf = nnf.repeat(2, axis=1).repeat(2, axis=2) * 2
        nnf[:, 1::2, :, 0] += 1
        nnf[:, :, 1::2, 1] += 1
        # check if scale is 2
        height, width = self.pyramid_heights[level], self.pyramid_widths[level]
        if height != nnf.shape[0] * 2 or width != nnf.shape[1] * 2:
            nnf_torch = torch.as_tensor(nnf, device='cuda', dtype=torch.float32)
            nnf_torch = nnf_torch.permute(0, 3, 1, 2)
            nnf_resized = F.interpolate(nnf_torch, size=(height, width), mode='bilinear', align_corners=False)
            nnf_resized = nnf_resized.permute(0, 2, 3, 1)
            nnf = cp.asarray(nnf_resized).astype(cp.int32)
            nnf = self.patch_matchers[level].clamp_bound(nnf)
        return nnf

    def apply_nnf_to_image(self, nnf, image):
        """Apply nnf to image.

        Args:
            nnf: The nnf.
            image: The image.
        """
        with cp.cuda.Device(self.gpu_id):
            image = self.patch_matchers[-1].pad_image(image)
            image = self.patch_matchers[-1].apply_nnf_to_image(nnf, image)
        return image

    def estimate_nnf(self, source_guide, target_guide, source_style):
        """Estimate nnf.

        Args:
            source_guide: The source guide.
            target_guide: The target guide.
            source_style: The source style.
        """
        with cp.cuda.Device(self.gpu_id):
            if not isinstance(source_guide, cp.ndarray):
                source_guide = cp.array(source_guide, dtype=cp.float32)
            if not isinstance(target_guide, cp.ndarray):
                target_guide = cp.array(target_guide, dtype=cp.float32)
            if not isinstance(source_style, cp.ndarray):
                source_style = cp.array(source_style, dtype=cp.float32)
            for level in range(self.pyramid_level):
                nnf = self.initialize_nnf(source_guide.shape[0]) if level==0 else self.update_nnf(nnf, level)
                source_guide_ = self.resample_image(source_guide, level)
                target_guide_ = self.resample_image(target_guide, level)
                source_style_ = self.resample_image(source_style, level)
                nnf, target_style = self.patch_matchers[level].estimate_nnf(
                    source_guide_, target_guide_, source_style_, nnf
                )
        return nnf.get(), target_style.get()
