# !/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import numpy as np
import random
import utils
import training

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


def ArchitectureSearch(grid, parameters, X, Y, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None, hp=False, embtp=None, sw=None, qn =False):


    mse_train = []
    mse_val = []
    mse_train_sd = []
    mse_val_sd = []

    for i in range(len(grid)):
        model_design = {"layersizes": grid[i]}
        print(grid)

        #if exp == 2:
        running_losses = training.train_cv(parameters, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp, embtp=embtp, sw=sw, qn =qn) # train model
        #else:
        #    running_losses = training.train(parameters, model_design, X, Y, "./data/", splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, embtp=embtp)

        print("SHAPE TRAIN LOSS IN CV: ")
        print(running_losses["train_loss"].shape)
        print("SHAPE MEAN TRAIN LOSS IN CV: ")
        print(np.mean(running_losses["train_loss"]).shape)
        
        mse_train.append(np.mean(running_losses["train_loss"]))
        mse_val.append(np.mean(running_losses["val_loss"]))
        mse_train_sd.append(np.std(running_losses["train_loss"]))
        mse_val_sd.append(np.std(running_losses["val_loss"]))
        print(f"fitted model {i}")
        print("FORM OF MEAN(Train_losses):")
        print("(Expected: list containing vector of length 5")
        print(mse_train)

    df = pd.DataFrame(grid)
    df["train_loss"] = mse_train
    df["val_loss"] = mse_val
    df["train_loss_sd"] = mse_train_sd
    df["val_loss_sd"] = mse_val_sd
    df["ind_mini"] = ((np.array(mse_val)**2 + np.array(mse_val_sd)**2)/2)
    print(df)
    print("Random architecture search best result:")
    #print(df.loc[[df["val_loss"].idxmin() & df["val_loss_sd"].idxmin()]])
    print(df.loc[df["ind_mini"].idxmin()])
    # layersizes = grid[df["val_loss"].idxmin() &  df["val_loss_sd"].idxmin()]
    layersizes = grid[df["ind_mini"].idxmin()]

    return layersizes, df



def HParSearchSpace(gridsize, reg=False, emb=False):
    grid = []
    for i in range(gridsize):
        if emb:
            learning_rate = random.choice([1e-1, 9e-2, 8e-2, 7e-2, 6e-2, 5e-2, 4e-2, 3e-2, 2e-2, 1e-2, 9e-3, 8e-3, 7e-3, 6e-3, 5e-3, 4e-3, 3e-3, 2e-3, 1e-3, 9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4])
            batchsize = random.choice([8, 16, 32, 64, 128])
        else:

            learning_rate = random.choice(np.round(np.linspace(1e-6, 0.1),4))

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




def HParSearch(layersizes, grid, X, Y, splits, data, reg=None, emb = False, raw=None, res=None, ypreles=None, exp=None, hp=False, embtp=None, sw=None, qn=False):


    model_design = {"layersizes": layersizes}
    mse_train = []
    mse_val = []
    mse_train_sd = []
    mse_val_sd = []

    for i in range(len(grid)):
        if reg is not None:
            hparams = {"epochs": 100,
                       "batchsize": grid[i][1],
                       "lr": grid[i][0],
                       "eta": grid[i][2]
                       }
        else:
            # original epochs 200
            hparams = {"epochs": 100,
                       "batchsize": grid[i][1],
                       "lr": grid[i][0]
                       }
                       

        #if exp == 2:
        running_losses = training.train_cv(hparams, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp, embtp=embtp, sw=sw, qn=qn)
        #else:
        #    running_losses = training.train(hparams, model_design, X, Y, "./data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, embtp=embtp)

        mse_train.append(np.mean(running_losses["train_loss"]))
        mse_val.append(np.mean(running_losses["val_loss"]))
        mse_train_sd.append(np.std(running_losses["train_loss"]))
        mse_val_sd.append(np.std(running_losses["val_loss"]))
        #        mse_train.append(np.mean(np.mean(running_losses["train_loss"], axis=1)))
        #        mse_val.append(np.mean(np.mean(running_losses["val_loss"], axis=1)))
        #        mse_train_sd.append(np.mean(np.std(running_losses["train_loss"], axis=1)))
        #        mse_val_sd.append(np.mean(np.std(running_losses["val_loss"], axis=1)))        

        print(f"fitted model {i}")
        
    df = pd.DataFrame(grid)
    df["train_loss"] = mse_train
    df["val_loss"] = mse_val
    print("VAL_loss:")
    print(mse_val)
    df["train_loss_sd"] = mse_train_sd
    df["val_loss_sd"] = mse_val_sd
    df["ind_mini"] = ((np.array(mse_val)**2 + np.array(mse_val_sd)**2)/2)
    #print(df.loc[[df["val_loss"].idxmin() & df["val_loss_sd"].idxmin()]])                       

    print("For architecture:")
    print(layersizes)
    print(df.head())
    print("Random hparams search best result:")
    #print(df.loc[[df["val_loss"].idxmin() & df["val_loss_sd"].idxmin()]])
    print(df.loc[df["ind_mini"].idxmin()])
    hparams = grid[df["ind_mini"].idxmin()]
    # hparams = grid[df["val_loss"].idxmin() &  df["val_loss_sd"].idxmin()]
    print("Dataframe:", df)
    return hparams, df

