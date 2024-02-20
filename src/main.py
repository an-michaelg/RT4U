# main training loop for single-iteration training

"""
Main
-Process the yml config file
-Create an agent instance
-Train the agent with label evolution logic
"""
import os
from omegaconf import DictConfig, OmegaConf
import hydra

import lightning.pytorch as pl
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from agent import Supervised
from datamodule_cifar import CIFAR_Q_DataModule
from datamodule_as import ASDataModule
from datamodule_tmed import TMED2_DataModule
from utils import resolve_save_dir


def agent_builder(agent_name, init_args_dict, save_dir):
    # if agent_name == "Supervised":
        # return Supervised(**init_args_dict, save_dir=save_dir)
    # elif agent_name == "SupervisedDenseFeatures":
        # return SupervisedDenseFeatures(**init_args_dict, save_dir=save_dir)
    # elif agent_name == "SupervisedPrototypes":
        # return SupervisedPrototypes(**init_args_dict, save_dir=save_dir)
    # else:
        # raise ValueError
    return Supervised(**init_args_dict, save_dir=save_dir)
    
def datamodule_builder(dataset_name, data_args_dict):
    if dataset_name == "CIFAR":
        return CIFAR_Q_DataModule(**data_args_dict)
    elif dataset_name == "AS":
        return ASDataModule(**data_args_dict)
    elif dataset_name == "TMED2":
        return TMED2_DataModule(**data_args_dict)
    else:
        raise ValueError(f"Got dataset_name == {dataset_name}")

# launch with something like python main.py --config-name=name_of_your_yaml
@hydra.main(version_base=None, config_path=".", config_name="config")
def main_no_cli(cfg):  # config file is loaded via yaml
    # instantiate the datamodule
    dm = datamodule_builder(cfg.dataset, cfg.data)

    # determine the save directory then create model checkpointing and logging
    root_save_dir = cfg.logger.init_args.save_dir
    experiment_name = cfg.logger.init_args.name
    if cfg.test_only:
        new_exp_name = experiment_name
        full_save_dir = os.path.join(root_save_dir, experiment_name)
    else:
        full_save_dir, new_exp_name = resolve_save_dir(root_save_dir, experiment_name)
    cfg.logger.init_args.name = new_exp_name  # quick switcharooni on the name
    logger = WandbLogger(**cfg.logger.init_args)
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoint, dirpath=full_save_dir)

    # save the configs
    OmegaConf.save(cfg, os.path.join(full_save_dir, "hydra_config.yaml"))

    # instantiate the lightningmodule
    model = agent_builder(cfg.model.agent_name, cfg.model.init_args, full_save_dir)

    trainer = pl.Trainer(**cfg.trainer, callbacks=[checkpoint_callback], logger=logger)
    if cfg.test_only:
        trainer.test(model, ckpt_path=cfg.ckpt_path, datamodule=dm)
    else:
        trainer.fit(model, ckpt_path=cfg.ckpt_path, datamodule=dm)
        trainer.test(model, ckpt_path=None, datamodule=dm)

if __name__ == "__main__":
    main_no_cli()
