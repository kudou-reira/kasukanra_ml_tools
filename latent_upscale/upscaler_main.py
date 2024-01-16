import sys, os

# Define the base path for external repositories
base_path = '/external_repos'

# Prepend the base path to each repository path
repos = ['taming-transformers', 'stable-diffusion', 'latent-diffusion']
full_paths = [os.path.join(base_path, repo) for repo in repos]

# Extend sys.path with these full paths
sys.path.extend(full_paths)

from utils import clean_prompt, format_filename, save_image, fetch, make_upscaler_model, download_from_huggingface, load_model_from_config 

import json
import numpy as np
import time
import re
import requests
import io
import hashlib
from subprocess import Popen

import torch
from torch import nn
import torch.nn.functional as F

from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm, trange
from functools import partial

from ldm.util import instantiate_from_config
import k_diffusion as K

from modules.clip_tokenizer_transform import CLIPTokenizerTransform
from modules.clip_embedder import CLIPEmbedder
from modules.cfg_upscaler import CFGUpscaler

@torch.no_grad()
def condition_up(prompts, tok_up, text_encoder_up):
    return text_encoder_up(tok_up(prompts))

@torch.no_grad()
def run(seed, input_image, device, batch_size, prompt, decoder, vae_model_840k, vae_model_560k, model_up, tok_up, text_encoder_up, noise_aug_level, guidance_scale, sampler, steps, eta, tol_scale, num_samples, SD_Q):
        timestamp = int(time.time())
        if not seed:
            print('No seed was provided, using the current time.')
            seed = timestamp
        print(f'Generating with seed={seed}')
        seed_everything(seed)

        uc = condition_up(batch_size * [""], tok_up, text_encoder_up)
        c = condition_up(batch_size * [prompt], tok_up, text_encoder_up)

        if decoder == 'finetuned_840k':
            vae = vae_model_840k
        elif decoder == 'finetuned_560k':
            vae = vae_model_560k

        # image = Image.open(fetch(input_file)).convert('RGB')
        image = input_image
        image = TF.to_tensor(image).to(device) * 2 - 1
        low_res_latent = vae.encode(image.unsqueeze(0)).sample() * SD_Q
        low_res_decoded = vae.decode(low_res_latent/SD_Q)

        [_, C, H, W] = low_res_latent.shape

        # Noise levels from stable diffusion.
        sigma_min, sigma_max = 0.029167532920837402, 14.614642143249512

        model_wrap = CFGUpscaler(model_up, uc, cond_scale=guidance_scale)
        low_res_sigma = torch.full([batch_size], noise_aug_level, device=device)
        x_shape = [batch_size, C, 2*H, 2*W]

        def do_sample(noise, extra_args):
            # We take log-linear steps in noise-level from sigma_max to sigma_min, using one of the k diffusion samplers.
            sigmas = torch.linspace(np.log(sigma_max), np.log(sigma_min), steps+1).exp().to(device)
            if sampler == 'k_euler':
                return K.sampling.sample_euler(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args)
            elif sampler == 'k_euler_ancestral':
                return K.sampling.sample_euler_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
            elif sampler == 'k_dpm_2_ancestral':
                return K.sampling.sample_dpm_2_ancestral(model_wrap, noise * sigma_max, sigmas, extra_args=extra_args, eta=eta)
            elif sampler == 'k_dpm_fast':
                return K.sampling.sample_dpm_fast(model_wrap, noise * sigma_max, sigma_min, sigma_max, steps, extra_args=extra_args, eta=eta)
            elif sampler == 'k_dpm_adaptive':
                sampler_opts = dict(s_noise=1., rtol=tol_scale * 0.05, atol=tol_scale / 127.5, pcoeff=0.2, icoeff=0.4, dcoeff=0)
                return K.sampling.sample_dpm_adaptive(model_wrap, noise * sigma_max, sigma_min, sigma_max, extra_args=extra_args, eta=eta, **sampler_opts)

        image_id = 0
        for _ in range((num_samples-1)//batch_size + 1):
            if noise_aug_type == 'gaussian':
                latent_noised = low_res_latent + noise_aug_level * torch.randn_like(low_res_latent)
            elif noise_aug_type == 'fake':
                latent_noised = low_res_latent * (noise_aug_level ** 2 + 1)**0.5
            extra_args = {'low_res': latent_noised, 'low_res_sigma': low_res_sigma, 'c': c}

            noise = torch.randn(x_shape, device=device)
            up_latents = do_sample(noise, extra_args)

            pixels = vae.decode(up_latents/SD_Q) # equivalent to sd_model.decode_first_stage(up_latents)
            pixels = pixels.add(1).div(2).clamp(0,1)

            # Save samples.
            for j in range(pixels.shape[0]):
                img = TF.to_pil_image(pixels[j])
                save_image(img, timestamp=timestamp, index=image_id, prompt=prompt, seed=seed)
                image_id += 1


if __name__ == "__main__":
    model_up = make_upscaler_model(fetch('https://models.rivershavewings.workers.dev/config_laion_text_cond_latent_upscaler_2.json'),
                               fetch('https://models.rivershavewings.workers.dev/laion_text_cond_latent_upscaler_2_1_00470000_slim.pth'))
    
    # sd_model_path = download_from_huggingface("CompVis/stable-diffusion-v-1-4-original", "sd-v1-4.ckpt")
    vae_840k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-mse-original", "vae-ft-mse-840000-ema-pruned.ckpt")
    vae_560k_model_path = download_from_huggingface("stabilityai/sd-vae-ft-ema-original", "vae-ft-ema-560000-ema-pruned.ckpt")

    cpu = torch.device("cpu")
    device = torch.device("cuda")

    # sd_model = load_model_from_config("stable-diffusion/configs/stable-diffusion/v1-inference.yaml", sd_model_path)
    vae_model_840k = load_model_from_config("/external_repos/latent-diffusion/models/first_stage_models/kl-f8/config.yaml", vae_840k_model_path)
    vae_model_560k = load_model_from_config("/external_repos/latent-diffusion/models/first_stage_models/kl-f8/config.yaml", vae_560k_model_path)

    # sd_model = sd_model.to(device)
    vae_model_840k = vae_model_840k.to(device)
    vae_model_560k = vae_model_560k.to(device)
    model_up = model_up.to(device)

    tok_up = CLIPTokenizerTransform()
    text_encoder_up = CLIPEmbedder(device=device)

    # Load configuration from a JSON file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Extracting parameters from the configuration
    prompt = config["prompt"]
    num_samples = config["num_samples"]
    batch_size = config["batch_size"]
    decoder = config["decoder"]
    guidance_scale = config["guidance_scale"]
    noise_aug_level = config["noise_aug_level"]
    noise_aug_type = config["noise_aug_type"]
    sampler = config["sampler"]
    steps = config["steps"]
    tol_scale = config["tol_scale"]
    eta = config["eta"]
    seed = config["seed"]
    SD_C = config["SD_C"]
    SD_F = config["SD_F"]
    SD_Q = config["SD_Q"]
    input_image_url = config["input_image_url"]

    if 'input_image' not in globals():
        # Set a demo image on first run.
        input_image = Image.open(fetch(input_image_url)).convert('RGB')

    run(
        seed=seed,
        input_image=input_image,
        device=device,
        batch_size=batch_size,
        prompt=prompt,
        decoder=decoder,
        vae_model_840k=vae_model_840k,
        vae_model_560k=vae_model_560k,
        model_up=model_up,
        tok_up=tok_up,
        text_encoder_up=text_encoder_up,
        noise_aug_level=noise_aug_level,
        guidance_scale=guidance_scale,
        sampler=sampler,
        steps=steps,
        eta=eta,
        tol_scale=tol_scale,
        num_samples=num_samples,
        SD_Q=SD_Q,
    )