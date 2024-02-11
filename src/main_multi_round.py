# main training loop and logic for label evolution

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
import utils
import wandb


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
        raise NotImplementedError # we are doing the CIFAR one first
        #return ASDataModule(**data_args_dict)
    else:
        raise ValueError

# launch with something like python main.py --config-name=name_of_your_yaml (w/o file extension)
@hydra.main(version_base=None, config_path=".", config_name="config")
def main_no_cli(cfg):  # config file is loaded via yaml
    # based on the number of evolution iterations Ne, we will have
    # Ne directories to save the model training results
    num_evolution_iters = cfg.num_evolution_iters + 1
    reset_weights_between_rounds = cfg.reset_weights_between_rounds
    
    # get the root of the save directory, desired experiment name
    root_save_dir = cfg.logger.init_args.save_dir
    experiment_name = cfg.logger.init_args.name
    
    test_only = cfg.test_only
    
    # obtain the datamodule object
    dm = datamodule_builder(cfg.dataset, cfg.data)
    
    # alternate flow for test only mode - use the existing folder
    if cfg.test_only:
        full_save_dir = os.path.join(root_save_dir, experiment_name)
        
        # instantiate callbacks
        logger = WandbLogger(**cfg.logger.init_args)
        checkpoint_callback = ModelCheckpoint(**cfg.checkpoint, dirpath=full_save_dir)
        
        # save the configs for future reference
        OmegaConf.save(cfg, os.path.join(full_save_dir, "hydra_config_test.yaml"))
        
        # instantiate the lightningmodule
        model = agent_builder(cfg.model.agent_name, cfg.model.init_args, full_save_dir)
        trainer = pl.Trainer(**cfg.trainer, callbacks=[checkpoint_callback], logger=logger)
        
        # test the model
        trainer.test(model, ckpt_path=cfg.ckpt_path, datamodule=dm)
        return
    
    # else perform the multiple-round training
    ckpt_path_from_last_round = None
    # generate the name of the folder to save training results into
    full_save_dir_base, new_exp_name_base = utils.resolve_save_dir(root_save_dir, experiment_name)
    for ne in range(num_evolution_iters):
        print(f"--- META: Start of evolution iteration {ne} ---")
        
        # if we are using >1 evolution iters, create sub-experiments for the evolution iter
        if num_evolution_iters > 1:
            full_save_dir = os.path.join(full_save_dir_base, "round" + str(ne))
            new_exp_name = new_exp_name_base + "_round" + str(ne)
            os.makedirs(full_save_dir)
            print(f"Directory created at {full_save_dir}")
        else:
            full_save_dir = full_save_dir_base
            new_exp_name = new_exp_name_base
            
        # there might be a scoping issue here with new_exp_name and full_save_dir here
        print(full_save_dir)
        print(new_exp_name)
        
        # the new experiment name is used by the logger
        cfg.logger.init_args.name = new_exp_name
        logger = WandbLogger(**cfg.logger.init_args)
        
        # the new save directory is used by the checkpoint callback, other params are the same
        checkpoint_callback = ModelCheckpoint(**cfg.checkpoint, dirpath=full_save_dir)
        
        # save the configs
        OmegaConf.save(cfg, os.path.join(full_save_dir, "hydra_config.yaml"))
        
        # instantiate the model with randomly initialized weights
        model = agent_builder(cfg.model.agent_name, cfg.model.init_args, full_save_dir)
        # if we are on >1 evolution iters, and we want to use previous model weights, load them
        if ne > 0 and reset_weights_between_rounds == False:
            print(f"--- META: Loading weights from previous epoch ---")
            model.load_only_weights(ckpt_path_from_last_round)

        trainer = pl.Trainer(**cfg.trainer, callbacks=[checkpoint_callback], logger=logger)
        
        # run the training and test procedures
        trainer.fit(model, ckpt_path=cfg.ckpt_path, datamodule=dm)
        trainer.test(model, ckpt_path=None, datamodule=dm)
        
        # get the history from this round of training and create pseudolabels
        prediction_history = model.get_prediction_history()
        print(f"--- META: Creating new pseudolabels ---")
        new_pseudolabels = utils.convert_history_to_pseudo(prediction_history)
        
        # save the pseudolabels into a file for future reference
        save_path = os.path.join(full_save_dir, "pseudo.csv")
        utils.save_pseudolabels(new_pseudolabels, save_path=save_path)
        
        # prepare the next round of training - both save path and pseudolabel
        print(f"--- META: Loading pseudolabels for next iteration ---")
        dm.set_pseudo(new_pseudolabels)
        if reset_weights_between_rounds == False:
            ckpt_path_from_last_round = os.path.join(full_save_dir, "last.ckpt")
        
        # we are using the wandb logger, reset the logger for the next iteration
        wandb.finish()
        

if __name__ == "__main__":
    main_no_cli()
