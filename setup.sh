#!/bin/sh

# pytorch previous versions for certain cudas, e.g. cuda 10.1 and 11.3, etc
#https://pytorch.org/get-started/previous-versions/
#pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch

<< DockerSetup :
DockerSetup
apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

<< LinuxSetupPytorchDocker :
when using the docker image: pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
LinuxSetupPytorchDocker
pip install --upgrade pip
pip install pandas wandb tqdm seaborn jupyter jupyterlab imageio scikit-image scikit-learn lightning
pip install hydra-core --upgrade
pip install "jsonargparse[signatures]"


<< Test :
Test
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)"
