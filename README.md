## Introduction
Repository for "Reliable Multi-View Learning with Conformal Prediction for Aortic Stenosis Classification in Echocardiography"
Includes code to train ResNet-18 and R(2+1)D models with the RT4U algorithm using the Pytorch Lightning framework
Includes code for conformal prediction algorithm and interpretation of results in jupyter notebooks

## Installation
### Docker
- Initialize a docker container running pytorch at https://hub.docker.com/r/pytorch/pytorch/tags. The majority of the code is ran using pytorch:1.12.1-cuda11.3-cudnn8-runtime
- Pytorch version may be changed depending on which CUDA version is on your local device
- refer to docker_setup for example
- Run setup.sh within the container to install relevant libraries

### Conda
- Configure a conda environment with a gpu-compatible pytorch version
- Install the list of files in setup.sh

## Preparing data
Data folders are assumed to be stored in the /data volume. To change this behavior, modify the src/*.yaml files
- The folder should be organized as follows for CIFAR within /data/cifar-10-batches-py/ (installing via torchvision.datasets.CIFAR10 is fine)
  - data_batch_[1..5]
  - metadata
- The folder should be organized as follows for TMED within /data/TMED/approved_users_only/
  - DEV[165/479/56] folders
  - dataset folders
  - csv files
- The folder should be organized as follows for the private AS dataset within /data/Aortic_Stenosis/as_tom/round2/
  - plax folder
  - psax folder
  - sheets folder
  - annotations-all.csv

## Usage
YAML files
- Configure settings and hyperparameters in the config_x.yaml files
- For training without initial checkpoint, set test_only to false and "ckpt_path" to null
- For training with checkpoint, set "ckpt_path" to the path to a checkpoint file
- For testing, set ckpt_path and then set test_only to true
- Files created during the run are stored under a newly created folder, the folder can be configured in the logger section of the YAML

Running
- Refer to main_multi_round.py for the training flow
- We use Hydra to integrate the yaml settings into the training process, refer to training_runs_x.sh for examples
  
Checklist before your first run
- Did you install relevant libraries in setup.py?
- Did you put the data in the relevant folder?
- Did you read the .yaml file and edit relevant fields?
- Do you understand where the output files from your training will go?
