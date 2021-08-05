# !/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import numpy as np
import random
import utils
import temb

def ArchitectureSearchSpace(input_size, output_size, grid_size, max_layers, emb=False):
    grid = []
    if emb:
        n = 2
    else:
        n = 1
    for i in range(grid_size):
        nets = []
        for nn in range(n):
            layer_size = []
            nlayers = random.randint(1, max_layers)
            for j in range(nlayers):
                size = random.choice([2, 4, 8, 16, 32, 64, 128, 256])
                layer_size.append(size)
            nets.append(layer_size)
        if emb and nets not in grid:
            grid.append(nets)
        elif not emb and layer_size not in grid:
            grid.append(layer_size)
    return grid

def ArchitectureSearch(grid, parameters, X, Y, splits, data, reg=None, emb=False, raw=None, mn =None, std = None):

    mse_train = []
    mse_val = []

    for i in range(len(grid)):
        model_design = {"layersizes": grid[i]}
        print(grid)
        running_losses = temb.train_cv(parameters, model_design, X, Y, "./data/" , splits, data, reg, emb, raw, mn, std) # train model
        mse_train.append(np.mean(running_losses["train_loss"]))
        mse_val.append(np.mean(running_losses["val_loss"]))
        print(f"fitted model {i}")

    df = pd.DataFrame(grid)
    df["train_loss"] = mse_train
    df["val_loss"] = mse_val
    print("Random architecture search best result:")
    print(df.loc[[df["val_loss"].idxmin()]])
    layersizes = grid[df["val_loss"].idxmin()]

    return layersizes, df


def HParSearchSpace(gridsize, reg=False):
    grid = []
    for i in range(gridsize):
        learning_rate = random.choice(np.round(np.linspace(0.0001, 0.1),5))
        batchsize = random.choice([2, 4, 8, 16, 32, 64])
        if reg is not False:
            r = random.choice(np.round(np.linspace(0.000, 1.000, 1000), 4))
            if [learning_rate, batchsize, r] not in grid:
                grid.append([learning_rate, batchsize, r])
        else:
            if [learning_rate, batchsize] not in grid:
                grid.append([learning_rate, batchsize])
    return grid



def HParSearch(layersizes, grid, X, Y, splits, data, reg=None, emb = False, raw=None, mn = None, std = None):

    model_design = {"layersizes": layersizes}
    mse_train = []
    mse_val = []

    for i in range(len(grid)):
        if reg is not None:
            hparams = {"epochs": 300,
                       "batchsize": grid[i][1],
                       "lr": grid[i][0],
                       "eta": grid[i][2]
                       }
        else:
            hparams = {"epochs": 300,
                       "batchsize": grid[i][1],
                       "lr": grid[i][0]
                       }
                       

        running_losses = temb.train_cv(hparams, model_design, X, Y, "./data/" , splits, data, reg, emb, raw, mn, std)
        mse_train.append(np.mean(running_losses["train_loss"]))
        mse_val.append(np.mean(running_losses["val_loss"]))
        print(f"fitted model {i}")

    df = pd.DataFrame(grid)
    df["train_loss"] = mse_train
    df["val_loss"] = mse_val
    print("Random hparams search best result:")
    print(df.loc[[df["val_loss"].idxmin()]])
    hparams = grid[df["val_loss"].idxmin()]
    print("Dataframe:", df)
    return hparams, df




