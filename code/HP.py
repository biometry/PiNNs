# !/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import numpy as np
import random
import utils
import training
import embtraining


def ArchitectureSearchSpace(grid_size, max_layers):
    grid = []
    for i in range(grid_size):
        layer_size = []
        nlayers = random.randint(1, max_layers)
        for j in range(nlayers):
            size = random.choice([2, 4, 8, 16, 32, 64, 128])
            layer_size.append(size)
        if layer_size not in grid:
            grid.append([[16, 16], layer_size])
    return grid

def ArchitectureSearch(grid, parameters, X, Y, Xn, Mn, std, pre):

    mae_train = []
    mae_val = []

    for i in range(len(grid)):
        model_design = {"layer_sizes": grid[i]}
        print(grid)
        running_losses = embtraining.train(parameters, model_design, X.to_numpy(), Y.to_numpy(), Xn.to_numpy(), mean=Mn, std=std, pre=pre) # train model
        mae_train.append(np.mean(running_losses["train_loss"]))
        mae_val.append(np.mean(running_losses["val_loss"]))
        print(f"fitted model {i}")

    df = pd.DataFrame(grid)
    df["train_loss"] = mae_train
    df["val_loss"] = mae_val
    print("Random architecture search best result:")
    print(df.loc[[df["val_loss"].idxmin()]])
    layersizes = grid[df["val_loss"].idxmin()]

    return layersizes


def HParSearchSpace(gridsize):
    grid = []
    for i in range(gridsize):
        learning_rate = random.choice(np.linspace(0.0001, 0.1))
        #batchsize = random.choice([2, 4, 8, 16, 32, 64, 128, 256, 365])
        #dropout = random.choice([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        if [learning_rate, 2] not in grid:
            grid.append([learning_rate, 2])# batchsize])
    return grid



def HParSearch(layersizes, grid, X, Y, Xn, Mn, std, pre):

    model_design = {"layer_sizes": layersizes}
    mae_train = []
    mae_val = []

    for i in range(len(grid)):
        hparams = {"epochs": 300,
                   "batchsize": grid[i][1],
                   "learningrate": grid[i][0],
                   #"dropout": grid[i][2],
                   "history": 1}

        running_losses = embtraining.train(hparams, model_design, X.to_numpy(), Y.to_numpy(), Xn.to_numpy(), mean=Mn, std=std, pre=pre)
        mae_train.append(np.mean(running_losses["train_loss"]))
        mae_val.append(np.mean(running_losses["val_loss"]))
        print(f"fitted model {i}")

    df = pd.DataFrame(grid)
    df["train_loss"] = mae_train
    df["val_loss"] = mae_val
    print("Random hparams search best result:")
    print(df.loc[[df["val_loss"].idxmin()]])
    hparams = grid[df["val_loss"].idxmin()]

    return hparams




