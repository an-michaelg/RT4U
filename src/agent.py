# pytorch lightning module for training generic deep learning models

import time
import numpy as np
from tqdm import tqdm
import math
import pandas as pd
import os

import torch
import lightning.pytorch as pl
import torchmetrics

from model_builder import get_backbone
from loss import CELossAbstain
from utils import plot_emb, save_csv


class Supervised(pl.LightningModule):
    def __init__(
        self,
        backbone="r2plus1d_18",
        pretrained=True,
        num_classes=4,
        learning_rate=1e-4,
        save_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes

        """ we pass in save_dir to help us with locating the model
        and for saving intermediate outputs like TSNE
        For the lightning purists, I acknowledge that this
        really should be done with Callbacks - M
        """
        self.save_dir = save_dir
        self.emb_dir = os.path.join(self.save_dir, "embeddings")
        self.csv_dir = os.path.join(self.save_dir, "csvs")
        os.makedirs(self.emb_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        # Initialize model
        self.encoder, embedding_dim = get_backbone(backbone, pretrained)
        self.decoder = torch.nn.Linear(embedding_dim, self.prediction_dim)

        # this loss can be used with both class probabilities and integer class labels
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")

        # Define metrics for each stage and save intermediate outputs
        """
        Strictly speaking, the flow to save embeddings for train epoch
        should be to train the epoch with gradients, then gradient off
        and run the images through training set again to get embeddings
        but it would take more time. Here we are a bit lazy and directly
        save the embeddings the model develop during the training epoch.
        For visualization purposes, this is faster but cause earlier epochs
        to look more chaotic
        """
        metrics = {}
        self.cache = {}
        self.modes = ["train", "val", "test"]
        for j in self.modes:
            metrics[j + "_f1"] = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            metrics[j + "_acc"] = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.cache[j + "_z"] = []
            self.cache[j + "_y"] = []
            self.cache[j + "_filename"] = []
            self.cache[j + "_pred"] = []
        self.metrics = torch.nn.ModuleDict(metrics)

        # Define optimizer and scheduler
        self.lr = learning_rate

    def forward(self, x):
        z = self.encoder(x)
        logits = self.decoder(z)
        return {"logits": logits, "z": z}

    def loss_wrapper(self, logits, y):
        return self.ce_loss(logits, y)

    def common_step(self, batch, batch_idx, mode="train"):
        x = batch["cine"]
        y = batch["target_AS"]
        if self.label_mode == "pseudo":
            pseudo = batch["target_pseudo"]
            
        outs = self.forward(x)

        # compute losses
        if mode == 'train' and self.label_mode == 'pseudo':
            loss = self.loss_wrapper(outs["logits"], pseudo)
        else:
            loss = self.loss_wrapper(outs["logits"], y)  # self.ce_loss(outs["logits"], y)

        # update metrics
        acc = self.metrics[mode + "_acc"](outs["logits"][:, : self.num_classes], y)
        f1 = self.metrics[mode + "_f1"](outs["logits"][:, : self.num_classes], y)

        log = {mode + "_loss": loss, mode + "_acc": acc, mode + "_f1": f1}
        self.log_dict(log)
        self.cache[mode + "_z"].append(outs["z"].detach().cpu())
        self.cache[mode + "_y"].append(y.cpu())
        self.cache[mode + "_filename"].extend(batch["filename"])
        self.cache[mode + "_pred"].append(outs["logits"].detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx, "test")
        return loss

    def predict_step(self, batch, batch_idx):
        # reveal all relevant input information
        x = batch["cine"]
        outs = self.forward(x)
        batch.update(outs)
        return batch  # , logits, z, attn

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def plot_emb_wrapper(self, z, y, fig_path, title):
        if len(z) > 50:  # define minimum number of points for plotting
            plot_emb(
                z,
                y,
                fig_path,
                protos=None,
                y_protos=None,
                title=title,
                compression="tsne",
            )

    def common_epoch_end(self, mode="train"):
        # print the mode-related metrics and log them
        acc_epoch = self.metrics[mode + "_acc"].compute()
        f1_epoch = self.metrics[mode + "_f1"].compute()
        print(f"{mode}: acc_epoch: {acc_epoch}, f1_epoch: {f1_epoch}")
        self.log_dict({mode + "_acc_epoch": acc_epoch, mode + "_f1_epoch": f1_epoch})

        # reset the metric in question
        self.metrics[mode + "_acc"].reset()
        self.metrics[mode + "_f1"].reset()

        # plot the embedding visualization
        z_saved = torch.cat(self.cache[mode + "_z"]).numpy()
        y_saved = torch.cat(self.cache[mode + "_y"]).numpy()
        title = f"{mode}_{self.current_epoch}"
        emb_save_name = title + ".jpg"
        emb_save_path = os.path.join(self.emb_dir, emb_save_name)
        self.plot_emb_wrapper(z_saved, y_saved, emb_save_path, title)

        # save a copy of the cached predictions
        pred_saved = torch.cat(self.cache[mode + "_pred"]).numpy()
        filename_saved = self.cache[mode + "_filename"]
        csv_save_name = title + ".csv"
        csv_save_path = os.path.join(self.csv_dir, csv_save_name)
        save_csv(filename_saved, y_saved, pred_saved, csv_save_path)

        for k in self.cache.keys():
            if mode in k:
                self.cache[k].clear()

    def on_train_epoch_end(self):
        self.common_epoch_end("train")

    def on_validation_epoch_end(self):
        self.common_epoch_end("val")

    def on_test_epoch_end(self):
        self.common_epoch_end("test")