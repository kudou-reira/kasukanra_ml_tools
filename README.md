# Project Overview
The purpose of these experiments is to establish a baseline of image to caption. Basically, we are looking for either classification labeling or image description using a lightweight transformer.

# Relevant Resources


# Dataset Images


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

After that, run one of the files.

Example:
```
CUDA_VISIBLE_DEVICES=0 python blip2_vanilla_grid.py
```