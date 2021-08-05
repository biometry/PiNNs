# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import trainloaded
import embtraining
import torch
import pandas as pd
import numpy as np

x, y, mn, std, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
splits = len(x.index.year.unique())
print(splits)
x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 150, 4)

# architecture search
layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 300, 'batchsize': 8, 'lr':0.01}, x, y, splits, "arSmlp")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(150)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpmlp")

print( 'hyperparameters: ', hpars)


grid.to_csv("./NmlpHP.csv")
argrid.to_csv("./NmlpAS.csv")
