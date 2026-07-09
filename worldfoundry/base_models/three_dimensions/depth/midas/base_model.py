"""Module for base_models -> three_dimensions -> depth -> midas -> base_model.py functionality."""

import torch


class BaseModel(torch.nn.Module):
    """Base model implementation."""
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device('cpu'))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)
