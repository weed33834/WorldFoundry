"""Module for base_models -> diffusion_model -> diffsynth -> extensions -> FastBlend -> runners -> accurate.py functionality."""

from ..patch_match import PyramidPatchMatcher
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


class AccurateModeRunner:
    """Accurate mode runner implementation."""
    def __init__(self):
        """Init."""
        pass

    def run(self, frames_guide, frames_style, batch_size, window_size, ebsynth_config, desc="Accurate Mode", save_path=None):
        """Run.

        Args:
            frames_guide: The frames guide.
            frames_style: The frames style.
            batch_size: The batch size.
            window_size: The window size.
            ebsynth_config: The ebsynth config.
            desc: The desc.
            save_path: The save path.
        """
        patch_match_engine = PyramidPatchMatcher(
            image_height=frames_style[0].shape[0],
            image_width=frames_style[0].shape[1],
            channel=3,
            use_mean_target_style=True,
            **ebsynth_config
        )
        # run
        n = len(frames_style)
        for target in tqdm(range(n), desc=desc):
            l, r = max(target - window_size, 0), min(target + window_size + 1, n)
            remapped_frames = []
            for i in range(l, r, batch_size):
                j = min(i + batch_size, r)
                source_guide = np.stack([frames_guide[source] for source in range(i, j)])
                target_guide = np.stack([frames_guide[target]] * (j - i))
                source_style = np.stack([frames_style[source] for source in range(i, j)])
                _, target_style = patch_match_engine.estimate_nnf(source_guide, target_guide, source_style)
                remapped_frames.append(target_style)
            frame = np.concatenate(remapped_frames, axis=0).mean(axis=0)
            frame = frame.clip(0, 255).astype("uint8")
            if save_path is not None:
                Image.fromarray(frame).save(os.path.join(save_path, "%05d.png" % target))
