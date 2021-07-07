# !/usr/bin/env python
# coding: utf-8

import torch
import pandas as pd
import numpy as np
import random
import utils
import embtraining

def ArchitectureSearchSpace(input_size, output_size, grid_size, max_layers):
    grid = []
        
    for i in range(grid_size):
        pair = []
        for j in range(2):
            layer_size = []
            nlayers = random.randint(1, max_layers)
            for j in range(nlayers):
                size = random.choice([2,4,8,16,32,64,128])
                layer_size.append(size)
            pair.append(layer_size)
        if pair not in grid:
            grid.append(pair)

    return grid

def ArchitectureSearch(grid, parameters, X, Y, Xn):

    mae_train = []
    mae_val = []

    for i in range(len(grid)):
        model_design = {"layer_sizes": grid[i]}
        running_losses = embtraining.train(parameters, model_design, X.to_numpy(),Y.to_numpy(), Xn)
        mae_train.append(np.mean(running_losses["train_loss"]))
        mae_val.append(np.mean(running_losses["val_loss"]))



    df = pd.DataFrame(grid)
    df["train_loss"] = mae_train
    df["val_loss"] = mae_val
    print("Best Result: ")
    print(df.loc[[df["val_loss"].idxmin()]])
    layersizes = grid[df["val_loss"].idxmin()]

    return layersizes


def HParSearchSpace(gridsize):
    grid = []
    for i in range(gridsize):
        learning_rate = random.choice([0.01, 0.03, 0.05, 0.001, 0.003, 0.005])
        batch_size = random.choice([2, 4, 8, 16, 32, 64, 128])
        if [learning_rate, batch_size] not in grid:
            grid.append([learning_rate, batch_size])
    return grid


def HParSearch(layersizes, grid, X, Y, Xn):

    model_design = {"layer_sizes": layersizes}
    mae_train = []
    mea_val = []

    for i in range(len(grid)):
        hparams = {"epochs": 300,
                   "batchsize": grid[i][1],
                   "learningrate": grid[i][0],
                   "history": 0}

        running_loss = embtraining.train(hparams, model_design, X.to_numpy(), Y.to_numpy(), Xn)
        mae_train.append(np.mean(running_loss["train_loss"]))
        mae_val.append(np.mean(running_loss["val_loss"]))

        df = pd.DataFrame(grid)
        df["train_loss"] = mae_train
        df["val_loss"] = mae_val
        print("Random hparams best: ")
        print(df.loc[[df["val_loss"].idxmin()]])
        hparams = grid[df["val_loss"].idxmin()]

        return hparams
    
