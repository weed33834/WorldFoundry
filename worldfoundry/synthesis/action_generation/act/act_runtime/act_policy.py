from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping


def kl_divergence(mu, logvar):
    """Compute the ACT CVAE KL loss components.

    Args:
        mu: Latent mean tensor.
        logvar: Latent log-variance tensor.
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return klds.sum(1).mean(0, True), klds.mean(0), klds.mean(1).mean(0, True)


def build_policy_class():
    """Create the ACT policy class after torch dependencies are available.

    Args:
        None.
    """
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torch.nn import functional as F

    from .models import build_ACT_model

    class ACTPolicy(nn.Module):
        def __init__(self, args: Any) -> None:
            super().__init__()
            self.model = build_ACT_model(args)
            self.kl_weight = float(getattr(args, "kl_weight", 10.0))

        def forward(self, qpos, image, actions=None, is_pad=None):
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = normalize(image)
            if actions is not None:
                actions = actions[:, : self.model.num_queries]
                is_pad = is_pad[:, : self.model.num_queries]
                a_hat, _is_pad_hat, (mu, logvar) = self.model(qpos, image, None, actions, is_pad)
                total_kld, _dim_wise_kld, _mean_kld = kl_divergence(mu, logvar)
                all_l1 = F.l1_loss(actions, a_hat, reduction="none")
                l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
                return {"l1": l1, "kl": total_kld[0], "loss": l1 + total_kld[0] * self.kl_weight}
            a_hat, _is_pad_hat, (_mu, _logvar) = self.model(qpos, image, None)
            return a_hat

    return ACTPolicy


def load_dataset_stats(checkpoint_dir: Path) -> Mapping[str, Any] | None:
    """Load ACT dataset normalization statistics when staged with weights.

    Args:
        checkpoint_dir: Directory expected to contain dataset_stats.pkl.
    """
    stats_path = checkpoint_dir / "dataset_stats.pkl"
    if not stats_path.is_file():
        return None
    with stats_path.open("rb") as handle:
        return pickle.load(handle)
