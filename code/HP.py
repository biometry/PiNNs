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
    print("AS Grid", len(grid))
    return grid

<<<<<<< HEAD
def ArchitectureSearch(grid, parameters, X, Y, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None, hp=False, embtp=None, sw=None):
=======
def ArchitectureSearch(grid, parameters, X, Y, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None, hp=False):
>>>>>>> origin/main

    mse_train = []
    mse_val = []

    for i in range(len(grid)):
        model_design = {"layersizes": grid[i]}
        print(grid)
<<<<<<< HEAD
        #if exp == 2:
        running_losses = temb.train_cv(parameters, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp, embtp=embtp, sw=sw) # train model
        #else:
        #    running_losses = temb.train(parameters, model_design, X, Y, "./data/", splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, embtp=embtp)
=======
        if exp == 2:
            running_losses = temb.train_cv(parameters, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp) # train model
        else:
            running_losses = temb.train(parameters, model_design, X, Y, "./data/", splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles)
>>>>>>> origin/main
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


def HParSearchSpace(gridsize, reg=False, emb=False):
    grid = []
    for i in range(gridsize):
        if emb:
            learning_rate = random.choice([1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4])
            batchsize = random.choice([8, 16, 32, 64, 128])
        else:
<<<<<<< HEAD
            learning_rate = random.choice(np.round(np.linspace(1e-6, 0.1),4))
=======
            learning_rate = random.choice(np.round(np.linspace(0.0001, 0.1),4))
>>>>>>> origin/main
            batchsize = random.choice([2, 4, 8, 16, 32, 64])
        if reg is not False:
            r = random.choice(np.round(np.linspace(0.000, 1.000, 200), 4))
            if [learning_rate, batchsize, r] not in grid:
                grid.append([learning_rate, batchsize, r])
        else:
            if [learning_rate, batchsize] not in grid:
                grid.append([learning_rate, batchsize])
    print('HP Grid', grid)
    return grid



<<<<<<< HEAD
def HParSearch(layersizes, grid, X, Y, splits, data, reg=None, emb = False, raw=None, res=None, ypreles=None, exp=None, hp=False, embtp=None, sw=None):
=======
def HParSearch(layersizes, grid, X, Y, splits, data, reg=None, emb = False, raw=None, res=None, ypreles=None, exp=None, hp=False):
>>>>>>> origin/main

    model_design = {"layersizes": layersizes}
    mse_train = []
    mse_val = []

    for i in range(len(grid)):
        if reg is not None:
            hparams = {"epochs": 500,
                       "batchsize": grid[i][1],
                       "lr": grid[i][0],
                       "eta": grid[i][2]
                       }
        else:
            hparams = {"epochs": 500,
                       "batchsize": grid[i][1],
                       "lr": grid[i][0]
                       }
                       
<<<<<<< HEAD
        #if exp == 2:
        running_losses = temb.train_cv(hparams, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp, embtp=embtp, sw=sw)
        #else:
        #    running_losses = temb.train(hparams, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, embtp=embtp)
=======
        if exp == 2:
            running_losses = temb.train_cv(hparams, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp)
        else:
            running_losses = temb.train(hparams, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles)
>>>>>>> origin/main
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




