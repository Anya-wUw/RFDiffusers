import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from torch.utils.data import  Dataset

import pandas as pd 
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers.optimization import Adafactor, AdafactorSchedule
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
import json
import diffusers
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.optimization import get_scheduler
from PIL import Image

from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPTextModelWithProjection
from schrodinger_scheduler import SchrodingerBridgeScheduler

import random

from diffusers import (
    AutoencoderKL,
    # DDPMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

# from omegaconf import OmegaConf
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from datasets import load_dataset
from flows.schedulers import FlowScheduler

import copy 

import os
import argparse
from tqdm import tqdm

logger = get_logger(__name__, log_level="INFO")

def get_time_embed(unet, sample, timestep):
    timesteps = timestep
    if not torch.is_tensor(timesteps):
        is_mps = sample.device.type == "mps"
        if isinstance(timestep, float):
            dtype = torch.float32 if is_mps else torch.float64
        else:
            dtype = torch.int32 if is_mps else torch.int64
        timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
    elif len(timesteps.shape) == 0:
        timesteps = timesteps[None].to(sample.device)

    timesteps = timesteps.expand(sample.shape[0])

    t_emb = unet.time_proj(timesteps)

    t_emb = t_emb.to(dtype=sample.dtype)
    
    return t_emb

def get_class_embed(unet, sample: torch.Tensor, class_labels):
    class_emb = None
    if unet.class_embedding is not None:
        if class_labels is None:
            raise ValueError("class_labels should be provided when num_class_embeds > 0")

        if unet.config.class_embed_type == "timestep":
            class_labels = unet.time_proj(class_labels)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # there might be better ways to encapsulate this.
            class_labels = class_labels.to(dtype=sample.dtype)

        class_emb = unet.class_embedding(class_labels).to(dtype=sample.dtype)
        return class_emb
    
    
    
def get_aug_embed(
    unet, emb: torch.Tensor, encoder_hidden_states: torch.Tensor, added_cond_kwargs
):
    aug_emb = None
    if unet.config.addition_embed_type == "text":
        aug_emb = unet.add_embedding(encoder_hidden_states)
    elif unet.config.addition_embed_type == "text_image":
        # Kandinsky 2.1 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{unet.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )

        image_embs = added_cond_kwargs.get("image_embeds")
        text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
        aug_emb = unet.add_embedding(text_embs, image_embs)
    elif unet.config.addition_embed_type == "text_time":
        # SDXL - style
        if "text_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{unet.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `text_embeds` to be passed in `added_cond_kwargs`"
            )
        text_embeds = added_cond_kwargs.get("text_embeds")
        if "time_ids" not in added_cond_kwargs:
            raise ValueError(
                f"{unet.__class__} has the config param `addition_embed_type` set to 'text_time' which requires the keyword argument `time_ids` to be passed in `added_cond_kwargs`"
            )
        time_ids = added_cond_kwargs.get("time_ids")
        time_embeds = unet.add_time_proj(time_ids.flatten())
        time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
        add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
        add_embeds = add_embeds.to(emb.dtype)
        aug_emb = unet.add_embedding(add_embeds)
    elif unet.config.addition_embed_type == "image":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs:
            raise ValueError(
                f"{unet.__class__} has the config param `addition_embed_type` set to 'image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        aug_emb = unet.add_embedding(image_embs)
    elif unet.config.addition_embed_type == "image_hint":
        # Kandinsky 2.2 - style
        if "image_embeds" not in added_cond_kwargs or "hint" not in added_cond_kwargs:
            raise ValueError(
                f"{unet.__class__} has the config param `addition_embed_type` set to 'image_hint' which requires the keyword arguments `image_embeds` and `hint` to be passed in `added_cond_kwargs`"
            )
        image_embs = added_cond_kwargs.get("image_embeds")
        hint = added_cond_kwargs.get("hint")
        aug_emb = unet.add_embedding(image_embs, hint)
    return aug_emb

    


def get_unet_middle_states(
    self,
    sample,
    timestep,
    encoder_hidden_states,
    class_labels=None,
    timestep_cond=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    added_cond_kwargs=None,
    encoder_attention_mask=None,
    return_dict=False,
    use_middle_states=False,
    use_upsample_states_num=None
):
    # By default samples have to be AT least a multiple of the overall upsampling factor.
    # The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
    # However, the upsampling interpolation output size can be forced to fit any upsampling size
    # on the fly if necessary.
    default_overall_up_factor = 2**self.num_upsamplers

    # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
    forward_upsample_size = False
    upsample_size = None

    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0:
            # Forward upsample size to force interpolation output size.
            forward_upsample_size = True
            break

    # ensure attention_mask is a bias, and give it a singleton query_tokens dimension
    # expects mask of shape:
    #   [batch, key_tokens]
    # adds singleton query_tokens dimension:
    #   [batch,                    1, key_tokens]
    # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
    #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
    #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
    if attention_mask is not None:
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        # convert mask into a bias that can be added to attention scores:
        #       (keep = +0,     discard = -10000.0)
        attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
        attention_mask = attention_mask.unsqueeze(1)

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None:
        encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    # 0. center input if necessary
    if self.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)
    aug_emb = None

    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None:
        if self.config.class_embeddings_concat:
            emb = torch.cat([emb, class_emb], dim=-1)
        else:
            emb = emb + class_emb

    aug_emb = self.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )
    if self.config.addition_embed_type == "image_hint":
        aug_emb, hint = aug_emb
        sample = torch.cat([sample, hint], dim=1)

    emb = emb + aug_emb if aug_emb is not None else emb

    if self.time_embed_act is not None:
        emb = self.time_embed_act(emb)

    encoder_hidden_states = self.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs
    )

    # 2. pre-process
    sample = self.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in self.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            # For t2i-adapter CrossAttnDownBlock2D
            additional_residuals = {}

            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
                **additional_residuals,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    if not use_middle_states:
        return sample

    # 4. mid
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = self.mid_block(sample, emb)

    if use_upsample_states_num is None:
        return sample

    # 5. up
    for i, upsample_block in enumerate(self.up_blocks[use_upsample_states_num]):
        is_final_block = i == len(self.up_blocks) - 1

        res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        # if we have not reached the final block and need to forward the
        # upsample size, we do it here
        if not is_final_block and forward_upsample_size:
            upsample_size = down_block_res_samples[-1].shape[2:]

        if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
                upsample_size=upsample_size,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                upsample_size=upsample_size,
            )

    return sample


def _get_variance(self, t, prev_t, predicted_variance=None, variance_type=None):
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] # if prev_t >= 0 else self.one
    current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

    # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
    # and sample from it to get previous sample
    # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
    variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

    # we always take the log of variance, so clamp it to ensure it's not 0
    variance = torch.clamp(variance, min=1e-20)

    if variance_type is None:
        variance_type = self.config.variance_type

    # hacks - were probably added for training stability
    if variance_type == "fixed_small":
        variance = variance
    # for rl-diffuser https://arxiv.org/abs/2205.09991
    elif variance_type == "fixed_small_log":
        variance = torch.log(variance)
        variance = torch.exp(0.5 * variance)
    elif variance_type == "fixed_large":
        variance = current_beta_t
    elif variance_type == "fixed_large_log":
        # Glide max_log
        variance = torch.log(current_beta_t)
    elif variance_type == "learned":
        return predicted_variance
    elif variance_type == "learned_range":
        min_log = torch.log(variance)
        max_log = torch.log(current_beta_t)
        frac = (predicted_variance + 1) / 2
        variance = frac * max_log + (1 - frac) * min_log

    return variance

def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]

def ddpm_scheduler_step_batched(
    self,
    model_output,
    timestep,
    previous_timestep,
    sample,
    variance_noise,
    return_dict=True,
):
    t = timestep
    prev_t = previous_timestep

    if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = self.alphas_cumprod[t]
    alpha_prod_t_prev = self.alphas_cumprod[prev_t] # if prev_t >= 0 else self.one
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1 - current_alpha_t

    alpha_prod_t = unsqueeze_like(alpha_prod_t, sample).to(sample.dtype)
    alpha_prod_t_prev = unsqueeze_like(alpha_prod_t_prev, sample).to(sample.dtype)
    beta_prod_t = unsqueeze_like(beta_prod_t, sample).to(sample.dtype)
    beta_prod_t_prev = unsqueeze_like(beta_prod_t_prev, sample).to(sample.dtype)
    current_alpha_t = unsqueeze_like(current_alpha_t, sample).to(sample.dtype)
    current_beta_t = unsqueeze_like(current_beta_t, sample).to(sample.dtype)

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

    # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
    current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

    # 5. Compute predicted previous sample µ_t
    # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
    pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

    # 6. Add noise
    if self.variance_type == "fixed_small_log":
        variance = _get_variance(self, t, prev_t, predicted_variance=predicted_variance)
        variance = unsqueeze_like(variance, variance_noise).to(variance_noise.dtype) * variance_noise
    elif self.variance_type == "learned_range":
        variance = _get_variance(self, t, prev_t, predicted_variance=predicted_variance)
        variance = torch.exp(0.5 * variance)
        variance = unsqueeze_like(variance, variance_noise).to(variance_noise.dtype) * variance_noise
    else:
        variance = (_get_variance(self, t, prev_t, predicted_variance=predicted_variance) ** 0.5)
        variance = unsqueeze_like(variance, variance_noise).to(variance_noise.dtype) * variance_noise
    variance[t==0] = 0.0

    pred_prev_sample = pred_prev_sample + variance

    if not return_dict:
        return (pred_prev_sample,)

    return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
    
    
def tokens_to_device(token_dict, device):
    for key in token_dict:
        token_dict[key] = token_dict[key].to(device)
    return token_dict

def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids

def encode_prompt(text_encoders, tokenizers, prompt, text_input_ids_list=None):
    prompt_embeds_list = []

    for i, text_encoder in enumerate(text_encoders):
        if tokenizers is not None:
            tokenizer = tokenizers[i]
            text_input_ids = tokenize_prompt(tokenizer, prompt)
        else:
            assert text_input_ids_list is not None
            text_input_ids = text_input_ids_list[i]

        prompt_embeds = text_encoder(
            text_input_ids.to(text_encoder.device),
            output_hidden_states=True,
        )

        # copying diffusers completely
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        bs_embed, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(prompt_embeds)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def dict_to_cuda(d, device):
    for key in d:
        d[key] = d[key].to(device)
    return d

def get_key_step(step, log_step_interval=50):
    step_d = step // log_step_interval
    return 'loss_' + str(step_d * log_step_interval) + '_' + str((step_d + 1) * log_step_interval)


def compute_vae_encodings(images, vae):
    pixel_values = torch.cat(images, dim=0)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)

    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
        
    model_input = model_input * vae.config.scaling_factor
    return model_input

class PairedImageDataset(Dataset):
    def __init__(self, dir1, dir2, resolution=1024, random_crop=False, no_hflip=False):
        self.dir1 = dir1
        self.dir2 = dir2

        # List all files in the directories
        self.images1 = [os.path.join(dir1, img) for img in os.listdir(dir1) if img.endswith(('.jpg', '.jpeg', '.png'))]
        self.images2 = [os.path.join(dir2, img) for img in os.listdir(dir2) if img.endswith(('.jpg', '.jpeg', '.png'))]

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(resolution) if random_crop else transforms.CenterCrop(resolution),
            transforms.Lambda(lambda x: x) if no_hflip else transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        # The length is the minimum of the two datasets
        return min(len(self.images1), len(self.images2))

    def __getitem__(self, idx):
        # Randomly select one image from each directory
        img1_path = random.choice(self.images1)
        img2_path = random.choice(self.images2)

        # Open and transform the images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2
        
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--output_dir', default='sdxl_rect_flow_0.16', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--train_batch_size', default=2, type=int)
    parser.add_argument('--logging_dir', default='logs', type=str)
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int)
    parser.add_argument('--checkpoints_total_limit', default=None, type=int)
    parser.add_argument('--mixed_precision', default='no', type=str)
    parser.add_argument('--report_to', default='tensorboard', type=str)
    parser.add_argument('--seed', default=42, type=int)
    
    parser.add_argument('--scheduler', default='model/sdxl/scheduler', type=str)
    parser.add_argument('--tokenizer', default='model/sdxl/tokenizer', type=str)
    parser.add_argument('--tokenizer_2', default='model/sdxl/tokenizer_2', type=str)
    parser.add_argument('--vae', default='model/sdxl/vae', type=str)
    parser.add_argument('--text_encoder', default='model/sdxl/text_encoder', type=str)
    parser.add_argument('--text_encoder_2', default='model/sdxl/text_encoder_2', type=str)
    parser.add_argument('--unet', default='model/sdxl/unet', type=str)
    
    parser.add_argument('--lr', default= 5e-5, type=float)
    # parser.add_argument('--lr', default=1e-5, type=float)
    # parser.add_argument('--adam_beta1', default=0.9, type=float)
    # parser.add_argument('--adam_beta2', default=0.98, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_warmup_steps', default=0, type=int)
    parser.add_argument('--lr_scheduler', default='constant_with_warmup', type=str)
    parser.add_argument('--project_name', default='dpo_pick-a-pic', type=str)
    
    parser.add_argument('--resume_from_checkpoint', default='latest', type=str)
    parser.add_argument('--max_train_steps', default=1000, type=int)
    parser.add_argument('--checkpointing_steps', default=10, type=int)
    
    parser.add_argument('--max_grad_norm', default=1, type=int)
    
    parser.add_argument('--memory_efficient_dpo', default=False, type=bool)

    parser.add_argument('--use_lora', default=True, type=bool)
    parser.add_argument('--lora_rank', default=64, type=int)
    
    parser.add_argument('--allow_tf32', default=False, type=bool)
    # args = OmegaConf.load(parser.parse_args().config_path)
    
    args = parser.parse_args()
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    # accumulation_steps = 512
    
    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, project_dir=args.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config
    )
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
        
    if args.seed is not None:
        set_seed(1337)
        
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32        
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
        
    
    base_path = "stabilityai/stable-diffusion-xl-base-1.0"
    
    tokenizer = CLIPTokenizer.from_pretrained(f"{base_path}", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained(f"{base_path}", subfolder='tokenizer_2')
    
    text_encoder = CLIPTextModel.from_pretrained(f"{base_path}", subfolder='text_encoder')
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(f"{base_path}", subfolder='text_encoder_2')
    
    vae = AutoencoderKL.from_pretrained(f"{base_path}", subfolder='vae')
    
    unet = UNet2DConditionModel.from_pretrained(f"{base_path}", subfolder='unet')

    # noise_scheduler = SchrodingerBridgeScheduler(
    #                                     num_train_timesteps=1000,
    #                                     beta_start=0.0001,
    #                                     beta_end=0.02,
    #                                     timestep_spacing="leading"
    #                                 )

    noise_scheduler = FlowScheduler()
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    
    
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    text_encoders = [text_encoder, text_encoder_2]
    tokenizers = [tokenizer, tokenizer_2]
    
    
    # dataset = CustomPickAPic('../proj_dpo_shared/diffusion_stuff/abs_winners_pick.csv', '../proj_dpo_shared/diffusion_stuff/pick-a-pic-dataset')
    
    dataset = PairedImageDataset('img_align_celeba/img_align_celeba', 'safebooru_jpeg')
    
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    overrode_max_train_steps = False
    
    if args.use_lora:
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        unet.requires_grad_(False)
        unet.add_adapter(unet_lora_config)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.lr,
        # betas=(args.adam_beta1, args.adam_beta2),
        # weight_decay=args.weight_decay,
    )
    # optimizer = Adafactor(unet.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr, weight_decay=0)
    global_step = 0
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
#     print('SAVING INITIAL IMGS')
#     pipe = StableDiffusionXLPipeline(
#                         vae=vae,
#                         text_encoder=text_encoder,
#                         text_encoder_2=text_encoder_2,
#                         tokenizer=tokenizer,
#                         tokenizer_2=tokenizer_2,
#                         unet=unet,
#                         # scheduler=EulerAncestralDiscreteScheduler.from_pretrained(args.scheduler)
#                         scheduler=DPMSolverMultistepScheduler.from_pretrained(args.scheduler)
#                     )
    
#     generator = torch.manual_seed(69)
#     img_list = pipe(prompt=test_prompts,
#                     generator=generator
#                    ).images
    
#     os.makedirs(args.output_dir + '_imgs', exist_ok=True)
#     if not args.use_lora:
#         save_path = os.path.join(args.output_dir + '_imgs', f'checkpoint-{global_step}')
#     else:
#         save_path = os.path.join(args.output_dir+ '_imgs', f'checkpoint_LORA_rank_{args.lora_rank}-{global_step}')
#     os.makedirs(save_path, exist_ok=True)
#     for idx, img in enumerate(img_list):
#         img.save(os.path.join(save_path, f'{idx}.jpeg'))
                        
        
    unet, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(
        unet, optimizer, lr_scheduler, train_dataloader
    )
    
    # if args.training_config.use_ema:
    #     ema_model = EMA(unet, 0.9999)
    #     ema_model.to(accelerator.device)
    #     ema_model.requires_grad_(False)
        
    if accelerator.is_main_process:
        accelerator.init_trackers(args.project_name, {'testing': 1})
    
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info('***** Starting training *****')
    logger.info(f'With batch_size = {args.batch_size}')
    logger.info(f'Total train batch size = {total_batch_size}')
    

    
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            
            # if args.training_config.use_ema:
            #     ema_model.load_state_dict(torch.load(os.path.join(args.training_config.output_dir, path, 'ema.bin'), map_location='cpu'))
            global_step = int(path.split("-")[1])

    print('global_step =', global_step)
    
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    progress_bar.set_description("SDXL TRAINING BRRR")

    unet.train()
    train_loss = 0.0
    losses = []
    steps = []
    total_logs = []
        
    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.diffusion.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()
    
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(unet):
            
            real, anime = batch[0].to(weight_dtype).to(accelerator.device), batch[0].to(weight_dtype).to(accelerator.device)
            
            vae_output = compute_vae_encodings([real, anime], vae)
            real_latent, anime_latent = vae_output[:args.batch_size], vae_output[args.batch_size:]

            bsz = real_latent.shape[0]

            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (real_latent.shape[0],), device=accelerator.device)

            steps += list(torch.clone(timesteps).cpu().numpy())

            #flow matching case
            noisy_model_input = noise_scheduler.add_noise(anime_latent, real_latent, timesteps)

            target = noise_scheduler.get_velocity(anime_latent, real_latent, timesteps)
            
            
            def compute_time_ids():
                # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                target_size = (1024, 1024)
                add_time_ids = list((1024, 1024) + (0, 0) + target_size)
                add_time_ids = torch.tensor([add_time_ids])
                add_time_ids = add_time_ids.to(unet.device, dtype=torch.float32)
                return add_time_ids
            prompts = ['']*bsz
            
            add_time_ids = torch.cat(
                [compute_time_ids() for s in range(len(prompts))]
            ).to(unet.device)


            unet_added_conditions = {"time_ids": add_time_ids}
            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, tokenizers, prompts)

            unet_added_conditions.update({"text_embeds": pooled_prompt_embeds})

            model_pred = unet(
                noisy_model_input,
                timesteps,
                prompt_embeds,
                added_cond_kwargs=unet_added_conditions,
            ).sample

            bs, c, h, w = real_latent.shape

                
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean()
         
            losses += [loss.cpu().detach().item()] * args.batch_size


            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps

            accelerator.backward(loss)
                
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
        if accelerator.sync_gradients:
            # if args.training_config.use_ema:
            #     ema_model(unet)
            
            progress_bar.update(1)
            global_step += 1
            log_step_interval = 50
            logs_keys = [get_key_step(i) for i in range(0, noise_scheduler.config.num_train_timesteps, log_step_interval)]
            logs_dict = {key:[] for key in logs_keys}
            for i in range(len(steps)):
                logs_dict[get_key_step(steps[i])].append(losses[i])
            filtered_logs_keys = [get_key_step(i) for i in range(0, 1000, log_step_interval) if len(logs_dict[get_key_step(i)]) > 0]
            filtered_logs_dict = {key:float(np.array(logs_dict[key]).mean()) for key in filtered_logs_keys}
            filtered_logs_dict["train_loss"] = train_loss
            accelerator.log(filtered_logs_dict, step=global_step)
            train_loss = 0.0
            losses, steps = [], []
            
            if global_step % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    if not args.use_lora:
                        save_path = os.path.join(args.output_dir, f'checkpoint-{global_step}')
                    else:
                        save_path = os.path.join(args.output_dir, f'checkpoint_LORA_rank_{args.lora_rank}-{global_step}')
                    accelerator.save_state(save_path)
                    # if args.use_ema:
                    #     ema_path = os.path.join(save_path, 'ema.bin')
                    #     torch.save(ema_model.state_dict(), ema_path)
                    if args.use_lora:
                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                                get_peft_model_state_dict(unwrapped_unet)
                            )
                        StableDiffusionPipeline.save_lora_weights(
                                save_directory=save_path,
                                unet_lora_layers=unet_lora_state_dict,
                                safe_serialization=True,
                            )
                    
                    logger.info(f'Saved state to {save_path}')
                    #SAVING VISUAL INFO
                    pipe = StableDiffusionXLPipeline(
                        vae=vae,
                        text_encoder=text_encoder,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer,
                        tokenizer_2=tokenizer_2,
                        unet=unwrapped_unet,
                        # scheduler=EulerAncestralDiscreteScheduler.from_pretrained(args.scheduler)
                        scheduler=noise_scheduler
                    )
                    

                    # generator = torch.manual_seed(69)
                    # img_list = pipe(prompt=test_prompts,
                    #                 generator=generator
                    #                ).images
                    # if not args.use_lora:
                    #     save_path = os.path.join(args.output_dir + '_imgs', f'checkpoint-{global_step}')
                    # else:
                    #     save_path = os.path.join(args.output_dir + '_imgs', f'checkpoint_LORA_rank_{args.lora_rank}-{global_step}')
                    # os.makedirs(save_path, exist_ok=True)
                    # for idx, img in enumerate(img_list):
                    #     img.save(os.path.join(save_path, f'{idx}.jpeg'))
                        
                    # logger.info(f'Saved imgs to {save_path}')
                    
                    

        logs = {'step_loss': loss.detach().item(), 'lr': lr_scheduler.get_last_lr()[0]}
            
        total_logs.append(logs)
        
        with open(f'logs_{args.output_dir}', 'w') as fout:
            json.dump(total_logs, fout)

        progress_bar.set_postfix(**logs)

        if global_step >= args.max_train_steps:
            break
            
#     if accelerator.is_main_process:
#         print('SAVING ALL STORED AND ACCUMULATED GRADIENTS !!!!!!')
#         gradients = {}
#         try:
#             for name, parameter in unet.named_parameters():
#                 if parameter.grad is not None:
#                     gradients[name] = parameter.grad.cpu().detach().numpy()
#                     # Save gradients as needed, e.g., using numpy or torch.save
#         except:
#             print('need to unwrap unet!')
#             unwrapped_unet = unwrap_model(unet)
            
#             for name, parameter in unwrapped_unet.named_parameters():
#                 if parameter.grad is not None:
#                     gradients[name] = parameter.grad.cpu().detach().numpy()
#                     # Save gradients as needed, e.g., using numpy or torch.save
                    
        # torch.save(gradients, "accumulated_gradients.pt")

                    
    accelerator.end_training()
    
# Entry point for the script
if __name__ == "__main__":
    main()
