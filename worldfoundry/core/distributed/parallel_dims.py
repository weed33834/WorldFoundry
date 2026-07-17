"""Parallel-dimension validation and DeviceMesh construction."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property

from torch.distributed.device_mesh import init_device_mesh

logger = logging.getLogger(__name__)


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )
        for degree in (dp_replicate, cp, tp, pp):
            assert degree >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, "dp_shard must be -1 or >= 1."
        if dp_shard < 0:
            logger.info(
                "dp_shard=-1; deriving it from world_size %s // %s.",
                self.world_size,
                dp_replicate * cp * tp * pp,
            )
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
            logger.info("dp_shard is set to %s.", dp_shard)
        assert dp_shard >= 1

        if dp_replicate * dp_shard * cp * tp * pp != self.world_size:
            self.dp_replicate = self.world_size // (dp_shard * cp * tp * pp)
            logger.warning(
                "Invalid parallel dims: dp_replicate(%s) * dp_shard(%s) * cp(%s) * tp(%s) * pp(%s) != WORLD_SIZE(%s).",
                dp_replicate,
                dp_shard,
                cp,
                tp,
                pp,
                self.world_size,
            )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for degree, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if degree > 1:
                dims.append(degree)
                names.append(name)

        logger.info("Building %s-D device mesh with %s, %s.", len(dims), names, dims)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=tuple(names))

        dp_mesh_dim_names = []
        dp_shard_cp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        if dp_cp_mesh_dim_names:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        logger.info("mesh: %s", mesh)
        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp


__all__ = ["ParallelDims"]
