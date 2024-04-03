#!/bin/sh

NAME=rt4u_torch112_cuda113

# Don't forget to change the --volume to link to your workspace/data folders!

<< DockerTags :
DockerTags
# link to pytorch docker hub  #https://hub.docker.com/r/pytorch/pytorch/tags
# for purang 28, used cuda 11.6 version
DOCKER_TAG=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# for others, used cuda 11.3 version
#DOCKER_TAG=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

<< DockerContainerBuild :
DockerContainerBuild
docker run -it -p 9999:9999 --ipc=host \
      --gpus device=ALL \
      --name=$NAME  \
      --volume=$HOME/workspace:/workspace \
	  --volume=/data:/data:ro \
      $DOCKER_TAG
