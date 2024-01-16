# Project Overview
The purpose of this repository is to create ML tools with regards to Stable Diffusion. Currently, latent upscaler from this [notebook](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4) is implemented. So, you can generate 1024 x 1024 pixel resolution images using Stable Diffusion 1.4 like so:

```
python generate_main.py --prompt "gaelle seguillon, krenz cushart, hyper realistic, low angle shot, wide lens, atmospheric perspective, golden hour, desaturated, split tone, futuristic, complex machinery, small crowds of people, terraforming, final fantasy xiv, aria the animation, venice" --seed 1214452132124
```

The current CLI command `generate_main.py` takes in the a `--prompt` and `--seed` argument.  

# Relevant Resources
[Upscaler Notebook](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4)

# Experimental Images
Please check the `examples images` for some sample images.

I've added some here for your viewing pleasure.

![Alt text](URL)

# How to use
Clone this repository to your local environment.

Build it with this command:
```
docker build -t kasukanra_ml_tools .
```

Once your image has been built, start the `docker-compose` with this command:

```
docker-compose -p [your_env] up
```

Assigning the -p flag (project flag) avoids clashing of networks when multiple instances of the same `docker-compose` are being run.

My version will be:

```
docker-compose -p yeo_env up
```

Once the container is running, go into the container with this sample command:


```
docker exec -it [your-env]-kasukanra_ml-app-1 bash
```

```
docker exec -it yeo_env-kasukanra_ml-app-1 bash
```

This command will be different for you depending on your -p flag name.

Once you are inside the `docker` container, activate the virtual environment:

```
source /venv/bin/activate
```

Example:
```
python generate_main.py --prompt "gaelle seguillon, krenz cushart, hyper realistic, low angle shot, wide lens, atmospheric perspective, golden hour, desaturated, split tone, futuristic, complex machinery, small crowds of people, terraforming, final fantasy xiv, aria the animation, venice" --seed 1214452132124
```