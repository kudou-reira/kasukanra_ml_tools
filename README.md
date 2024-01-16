# Project Overview
The purpose of this repository is to create ML tools with regards to Stable Diffusion. Currently, latent upscaler from this [notebook](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4) is implemented. So, you can generate 1024 x 1024 pixel resolution images using Stable Diffusion 1.4 like so:

```
python generate_main.py --prompt "gaelle seguillon, krenz cushart, hyper realistic, low angle shot, wide lens, atmospheric perspective, golden hour, desaturated, split tone, futuristic, complex machinery, small crowds of people, terraforming, final fantasy xiv, aria the animation, venice" --seed 1214452132124
```

The current CLI command `generate_main.py` takes in the a `--prompt` and `--seed` argument.  

# Relevant Resources
[Upscaler Notebook](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4)

# Experimental Images
Please check the [examples images](https://github.com/kudou-reira/kasukanra_ml_tools/tree/main/example_images) for more sample images.

I've added some here for your viewing pleasure.

![Alt text](https://github.com/kudou-reira/kasukanra_ml_tools/blob/main/example_images/1705389339-00-xenoblade%20landscape,%20flying%20alien%20monstrous%20beasts%20inspired%20by%20peter%20mohrbacher,%20golden%20hour,%20hypers%E2%80%A6.png?raw=true)

![Alt text](https://github.com/kudou-reira/kasukanra_ml_tools/blob/main/example_images/1705388695-00-gaelle%20seguillon,%20krenz%20cushart,%20hyper%20realistic,%20low%20angle%20shot,%20wide%20lens,%20atmospheric%20perspective%E2%80%A6.png?raw=true)

# How to use
Clone this repository to your local environment.

Build it with this command:
```
docker build -t kasukanra_ml_tools .
```

## Volume Mounts
For some reason, I wasn't able to have volume syncrhonization using relative paths. As a result, my [docker-compose file](https://github.com/kudou-reira/kasukanra_ml_tools/blob/main/docker-compose.yml#L8) is using absolute paths for the volume mounts. Make sure to change these volumes paths on your own local environment.

Mounting `/.cache` saves you the trouble of having to fetch/download models every time you start up the container.

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