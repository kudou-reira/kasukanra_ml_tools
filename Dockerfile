# Start from a CUDA 11.8 base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    build-essential \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libffi-dev \
    python-openssl

# Clean up the apt cache to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Add pyenv to PATH
ENV PATH="/root/.pyenv/bin:$PATH"

# Setup environment for pyenv
RUN echo 'eval "$(pyenv init --path)"' >> /etc/profile.d/pyenv.sh
RUN echo 'eval "$(pyenv init -)"' >> /etc/profile.d/pyenv.sh
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> /etc/profile.d/pyenv.sh

# Make bash execute scripts when starting
SHELL ["/bin/bash", "-c"]

# Create and activate a virtual environment
RUN source /etc/profile.d/pyenv.sh && \
    pyenv install 3.9.13 && \
    pyenv global 3.9.13 && \
    pip install --upgrade pip && \
    pip install virtualenv && \
    virtualenv /venv

# Create a directory for external repositories
RUN mkdir /external_repos

# Clone necessary repositories into /external_repos
RUN git clone https://github.com/CompVis/stable-diffusion /external_repos/stable-diffusion && \
    git clone https://github.com/CompVis/taming-transformers /external_repos/taming-transformers && \
    git clone https://github.com/CompVis/latent-diffusion /external_repos/latent-diffusion && \
    git clone https://github.com/huggingface/diffusers /external_repos/diffusers

# Replace the line in the Stable Diffusion file which is giving errors
RUN sed -i 's/from pytorch_lightning.utilities.distributed import rank_zero_only/from pytorch_lightning.utilities.rank_zero import rank_zero_only/' /external_repos/stable-diffusion/ldm/models/diffusion/ddpm.py

# Delete line 347 and 348 in /external_repos/stable-diffusion/scripts/txt2img.py, it gives errors
# link to line ref: https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py#L347C5-L348C24
RUN sed -i '347,348d' /external_repos/stable-diffusion/scripts/txt2img.py

# Extend sys in txt2img.py
RUN echo -e "import sys, os\n\n\
base_path = '/external_repos'\n\
repos = ['taming-transformers', 'stable-diffusion', 'latent-diffusion']\n\
full_paths = [os.path.join(base_path, repo) for repo in repos]\n\
sys.path.extend(full_paths)\n\n\
$(cat /external_repos/stable-diffusion/scripts/txt2img.py)" > /external_repos/stable-diffusion/scripts/txt2img.py

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Activate virtual environment and Install PyTorch 2.1.2 with CUDA 11.8 support
RUN source /venv/bin/activate && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Activate virtual environment and Install the remaining Python dependencies
# (assuming other dependencies are listed in requirements.txt)
RUN source /venv/bin/activate && \
    pip install -r /app/requirements.txt

RUN source /venv/bin/activate && \
    pip install -e /external_repos/diffusers

# The container does nothing by default and will just run bash to keep it alive
CMD ["tail", "-f", "/dev/null"]