"""Module for base_models -> diffusion_model -> diffsynth -> schedulers -> flow_match.py functionality."""

import torch, math



class FlowMatchScheduler():
    """Flow match scheduler implementation."""

    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003/1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
        exponential_shift=False,
        exponential_shift_mu=None,
        shift_terminal=None,
    ):
        """Init.

        Args:
            num_inference_steps: The num inference steps.
            num_train_timesteps: The num train timesteps.
            shift: The shift.
            sigma_max: The sigma max.
            sigma_min: The sigma min.
            inverse_timesteps: The inverse timesteps.
            extra_one_step: The extra one step.
            reverse_sigmas: The reverse sigmas.
            exponential_shift: The exponential shift.
            exponential_shift_mu: The exponential shift mu.
            shift_terminal: The shift terminal.
        """
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.exponential_shift = exponential_shift
        self.exponential_shift_mu = exponential_shift_mu
        self.shift_terminal = shift_terminal
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None, dynamic_shift_len=None, exponential_shift_mu=None):
        """Set timesteps.

        Args:
            num_inference_steps: The num inference steps.
            denoising_strength: The denoising strength.
            training: The training.
            shift: The shift.
            dynamic_shift_len: The dynamic shift len.
            exponential_shift_mu: The exponential shift mu.
        """
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        if self.exponential_shift:
            if exponential_shift_mu is not None:
                mu = exponential_shift_mu
            elif dynamic_shift_len is not None:
                mu = self.calculate_shift(dynamic_shift_len)
            else:
                mu = self.exponential_shift_mu
            self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
        else:
            self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.shift_terminal is not None:
            one_minus_z = 1 - self.sigmas
            scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
            self.sigmas = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
            self.training = True
        else:
            self.training = False


    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """Step.

        Args:
            model_output: The model output.
            timestep: The timestep.
            sample: The sample.
            to_final: The to final.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample


    def return_to_timestep(self, timestep, sample, sample_stablized):
        """Return to timestep.

        Args:
            timestep: The timestep.
            sample: The sample.
            sample_stablized: The sample stablized.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output


    def add_noise(self, original_samples, noise, timestep):
        """Add noise.

        Args:
            original_samples: The original samples.
            noise: The noise.
            timestep: The timestep.
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample


    def training_target(self, sample, noise, timestep):
        """Training target.

        Args:
            sample: The sample.
            noise: The noise.
            timestep: The timestep.
        """
        target = noise - sample
        return target


    def training_weight(self, timestep):
        """Training weight.

        Args:
            timestep: The timestep.
        """
        timestep_id = torch.argmin((self.timesteps - timestep.to(self.timesteps.device)).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


    def calculate_shift(
        self,
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ):
        """Calculate shift.

        Args:
            image_seq_len: The image seq len.
            base_seq_len: The base seq len.
            max_seq_len: The max seq len.
            base_shift: The base shift.
            max_shift: The max shift.
        """
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu
