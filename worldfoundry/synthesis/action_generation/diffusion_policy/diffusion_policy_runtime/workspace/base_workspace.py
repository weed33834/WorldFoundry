from __future__ import annotations

import dill
from omegaconf import OmegaConf


class BaseWorkspace:
    """Minimum checkpoint-loading base for Diffusion Policy inference.

    Official checkpoints serialize a Hydra workspace target. WorldFoundry keeps
    only the state-loading surface needed to reconstruct the policy module for
    `predict_action`; training, snapshotting, and checkpoint-writing helpers are
    intentionally not packaged.
    """

    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: str | None = None) -> None:
        self.cfg = cfg
        self._output_dir = output_dir

    def run(self) -> None:
        raise RuntimeError("Diffusion Policy training workspaces are not packaged in WorldFoundry.")

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs) -> None:
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload["pickles"].keys()

        for key, value in payload["state_dicts"].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload["pickles"]:
                self.__dict__[key] = dill.loads(payload["pickles"][key])
