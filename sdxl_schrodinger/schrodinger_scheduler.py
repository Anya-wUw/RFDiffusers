import torch
import numpy as np
from typing import Optional

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput

class SchrodingerBridgeOutput(BaseOutput):
    prev_sample: torch.FloatTensor
    pred_original_sample: torch.FloatTensor = None

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]
        
class SchrodingerBridgeScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = "cuda",
        timestep_spacing: str = "leading",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device)
        self.init_noise_sigma = 1
        

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        

        self.std_fwd = torch.sqrt(1 - self.alphas_cumprod)
        self.std_bwd = torch.sqrt(self.alphas_cumprod)
        

        self.mu_x0, self.mu_x1, self.var = self.compute_gaussian_product_coef()
        self.std_sb = torch.sqrt(self.var)

        self.timesteps = None
        self.step_index = 0
        
    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)
        
    def compute_label(self, step, x0, xt):
        """ equation 12 from paper """
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()
        
    def compute_gaussian_product_coef(self):
        var = 1 / (1 / self.std_fwd**2 + 1 / self.std_bwd**2)
        mu_x0 = var / self.std_fwd**2
        mu_x1 = var / self.std_bwd**2
        return mu_x0, mu_x1, var

    def set_timesteps(self, num_inference_steps: int, device=None, timesteps=None):
        self.num_inference_steps = num_inference_steps
        self.timesteps = torch.linspace(self.num_train_timesteps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)
        self.step_index = 0

    def step(self, model_output, timestep, sample, x1=None, return_dict=True):
        t = timestep
        prev_t = self.timesteps[self.step_index + 1] if self.step_index < len(self.timesteps) - 1 else 0
        
        pred_x0 = model_output
        xt = self.p_posterior(prev_t, t, sample, pred_x0)
        
        if x1 is not None:
            mask = self.get_mask(x1)  # Implement this based on your corruption method
            xt = (1 - mask) * x1 + mask * xt
        
        self.step_index += 1

        if not return_dict:
            return (xt,)

        return SchrodingerBridgeOutput(prev_sample=xt, pred_original_sample=pred_x0)

    def q_sample(self, step, x0, x1):
        """q(x_t | x_0, x_1)"""
        mu_x0 = self.mu_x0[step].view(-1, 1, 1, 1)
        mu_x1 = self.mu_x1[step].view(-1, 1, 1, 1)
        std_sb = self.std_sb[step].view(-1, 1, 1, 1)
        
        xt = mu_x0 * x0 + mu_x1 * x1 + std_sb * torch.randn_like(x0)
        return xt

    def p_posterior(self, prev_t, t, x_t, x0):
        """p(x_{t-1} | x_t, x_0)"""
        std_t = self.std_fwd[t]
        std_prev_t = self.std_fwd[prev_t]
        std_delta = torch.sqrt(std_t**2 - std_prev_t**2)
        
        mu_x0, mu_xt, var = self.compute_gaussian_product_coef_step(std_prev_t, std_delta)
        
        xt_prev = mu_x0 * x0 + mu_xt * x_t
        if prev_t > 0:
            xt_prev = xt_prev + torch.sqrt(var) * torch.randn_like(xt_prev)
        
        return xt_prev

    def compute_gaussian_product_coef_step(self, std1, std2):
        var = 1 / (1 / std1**2 + 1 / std2**2)
        mu1 = var / std1**2
        mu2 = var / std2**2
        return mu1, mu2, var

    def add_noise(self, original_samples, noise, timesteps):
        return self.q_sample(timesteps, original_samples, noise)

    def __len__(self):
        return self.num_train_timesteps

    def scale_model_input(self, sample: torch.FloatTensor, timestep: Optional[int] = None) -> torch.FloatTensor:
        return sample

    def get_mask(self, x1):

        return torch.ones_like(x1)
