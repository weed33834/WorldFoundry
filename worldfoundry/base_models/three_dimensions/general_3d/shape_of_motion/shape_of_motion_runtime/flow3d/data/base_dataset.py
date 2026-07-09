"""Module for base_models -> three_dimensions -> general_3d -> shape_of_motion -> shape_of_motion_runtime -> flow3d -> data -> base_dataset.py functionality."""

from abc import abstractmethod

import torch
from torch.utils.data import Dataset, default_collate


class BaseDataset(Dataset):
    """Base dataset implementation."""
    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        """Keyframe idcs.

        Returns:
            The return value.
        """
        return torch.arange(self.num_frames)

    @abstractmethod
    def get_w2cs(self) -> torch.Tensor: ...

    @abstractmethod
    def get_Ks(self) -> torch.Tensor: ...

    @abstractmethod
    def get_image(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_depth(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_mask(self, index: int) -> torch.Tensor: ...

    def get_img_wh(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_tracks_3d(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns 3D tracks:
            coordinates (N, T, 3),
            visibles (N, T),
            invisibles (N, T),
            confidences (N, T),
            colors (N, 3)
        """
        ...

    @abstractmethod
    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns background points:
            coordinates (N, 3),
            normals (N, 3),
            colors (N, 3)
        """
        ...

    @staticmethod
    def train_collate_fn(batch):
        """Train collate fn.

        Args:
            batch: The batch.
        """
        collated = {}
        for k in batch[0]:
            if k not in [
                "query_tracks_2d",
                "target_ts",
                "target_w2cs",
                "target_Ks",
                "target_tracks_2d",
                "target_visibles",
                "target_track_depths",
                "target_invisibles",
                "target_confidences",
            ]:
                collated[k] = default_collate([sample[k] for sample in batch])
            else:
                collated[k] = [sample[k] for sample in batch]
        return collated
