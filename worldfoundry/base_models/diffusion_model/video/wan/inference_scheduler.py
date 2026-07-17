"""Shared flow-matching schedule operations used by Wan inference runtimes."""

from __future__ import annotations

import torch


class InferenceFlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps: int = 100,
        *,
        num_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        extra_one_step: bool = False,
    ) -> None:
        self.num_timesteps = int(num_timesteps)
        self.shift = float(shift)
        self.sigma_max = float(sigma_max)
        self.sigma_min = float(sigma_min)
        self.extra_one_step = bool(extra_one_step)
        self.set_timesteps(num_inference_steps)

    def set_timesteps(
        self,
        num_inference_steps: int,
        denoising_strength: float = 1.0,
    ) -> None:
        sigma_start = self.sigma_min + (
            self.sigma_max - self.sigma_min
        ) * float(denoising_strength)
        count = int(num_inference_steps) + int(self.extra_one_step)
        sigmas = torch.linspace(sigma_start, self.sigma_min, count)
        if self.extra_one_step:
            sigmas = sigmas[:-1]
        self.sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        self.timesteps = self.sigmas * self.num_timesteps

    def _indices(self, timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
        timesteps = self.timesteps.to(device=device, dtype=torch.float64)
        values = timestep.reshape(-1).to(device=device, dtype=torch.float64)
        return torch.argmin(
            (timesteps.unsqueeze(0) - values.unsqueeze(1)).abs(),
            dim=1,
        )

    def sigma_at(
        self,
        timestep: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        indices = self._indices(timestep, reference.device)
        return self.sigmas.to(
            device=reference.device,
            dtype=reference.dtype,
        )[indices].reshape(-1, 1, 1, 1)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.sigma_at(timestep, noise)
        return ((1.0 - sigma) * original_samples + sigma * noise).type_as(noise)

    def flow_to_x0(
        self,
        flow: torch.Tensor,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = flow.dtype
        flow64 = flow.double()
        noisy64 = noisy.double()
        sigma = self.sigma_at(timestep, flow64)
        return (noisy64 - sigma * flow64).to(original_dtype)

    def flow_step(
        self,
        flow: torch.Tensor,
        noisy: torch.Tensor,
        timestep: torch.Tensor,
        next_timestep: torch.Tensor,
    ) -> torch.Tensor:
        sigma = self.sigma_at(timestep, noisy)
        next_sigma = self.sigma_at(next_timestep, noisy)
        return noisy + flow.to(noisy.dtype) * (next_sigma - sigma)


__all__ = ["InferenceFlowMatchScheduler"]
