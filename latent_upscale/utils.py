import os
import re
import hashlib
import requests
import subprocess

from PIL import Image
from modules.noise_level_upscaler import NoiseLevelAndTextConditionedUpscaler
import torch
import k_diffusion as K

from requests.exceptions import HTTPError
import huggingface_hub
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# Specify the default save location
save_location = 'stable-diffusion-upscaler/%T-%I-%P.png'

cpu = torch.device("cpu")

def clean_prompt(prompt):
    badchars = re.compile(r'[/\\]')
    prompt = badchars.sub('_', prompt)
    if len(prompt) > 100:
        prompt = prompt[:100] + '…'
    return prompt

def format_filename(timestamp, seed, index, prompt):
    string = save_location
    string = string.replace('%T', f'{timestamp}')
    string = string.replace('%S', f'{seed}')
    string = string.replace('%I', f'{index:02}')
    string = string.replace('%P', clean_prompt(prompt))
    return string

def save_image(image, **kwargs):
    filename = format_filename(**kwargs)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    image.save(filename)

def fetch(url_or_path):
    if url_or_path.startswith('http:') or url_or_path.startswith('https:'):
        _, ext = os.path.splitext(os.path.basename(url_or_path))
        cachekey = hashlib.md5(url_or_path.encode('utf-8')).hexdigest()
        cachename = f'{cachekey}{ext}'

        cache_dir = 'cache'
        tmp_dir = 'tmp'
        cache_path = os.path.join(cache_dir, cachename)
        tmp_path = os.path.join(tmp_dir, cachename)

        if not os.path.exists(cache_path):
            os.makedirs(tmp_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            
            # Fetch the content from URL
            response = requests.get(url_or_path)
            if response.status_code == 200:
                with open(tmp_path, 'wb') as file:
                    file.write(response.content)
                os.rename(tmp_path, cache_path)
            else:
                raise Exception(f"Failed to fetch {url_or_path}, status code {response.status_code}")

        return cache_path
    return url_or_path

def make_upscaler_model(config_path, model_path, pooler_dim=768, train=False, device='cpu'):
    config = K.config.load_config(open(config_path))
    model = K.config.make_model(config)
    model = NoiseLevelAndTextConditionedUpscaler(
        model,
        sigma_data=config['model']['sigma_data'],
        embed_dim=config['model']['mapping_cond_dim'] - pooler_dim,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    model.load_state_dict(ckpt['model_ema'])
    model = K.config.make_denoiser_wrapper(config)(model)
    if not train:
        model = model.eval().requires_grad_(False)
    return model.to(device)

def download_from_huggingface(repo, filename):
  while True:
    try:
      return huggingface_hub.hf_hub_download(repo, filename)
    except HTTPError as e:
      if e.response.status_code == 401:
        # Need to log into huggingface api
        huggingface_hub.interpreter_login()
        continue
      elif e.response.status_code == 403:
        # Need to do the click through license thing
        print(f'Go here and agree to the click through license on your account: https://huggingface.co/{repo}')
        input('Hit enter when ready:')
        continue
      else:
        raise e

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model = model.to(cpu).eval().requires_grad_(False)
    return model

def run_txt2img(script_path, prompt, config, ckpt, seed, n_iter, n_samples, ddim_steps, height, width, skip_grid=False, additional_args=None):
    # Base command
    command = ["python", script_path]

    # Required arguments
    args = [
        "--prompt", prompt,
        "--config", config,
        "--ckpt", ckpt,
        "--seed", str(seed),
        "--n_iter", str(n_iter),
        "--n_samples", str(n_samples),
        "--ddim_steps", str(ddim_steps),
        "--H", str(height),  # Image height
        "--W", str(width)    # Image width
    ]

    # Add the skip_grid argument if requested
    if skip_grid:
        args.append("--skip_grid")

    # Add any additional optional arguments
    if additional_args is not None:
        args.extend(additional_args)

    # Combine command and arguments
    cmd = command + args

    # Run the subprocess and capture output in real-time
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, universal_newlines=True) as process:
        for line in process.stdout:
            print(line, end='')  # Print each line of output in real-time

        process.communicate()  # Wait for the subprocess to finish

    # Check return code to see if there were any errors
    if process.returncode != 0:
        print("Errors occurred. Return code:", process.returncode)

def get_latest_image(directory):
    # List all files in the directory
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    
    # Filter out files that are not images (optional, based on file extensions)
    files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort files by modification time, newest first
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Return the first file in the list (newest file)
    return files[0] if files else None