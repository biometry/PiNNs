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

x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
splits = len(x.index.year.unique())
print(splits)
<<<<<<< HEAD
print(x, y)
=======
print(y, y.to_frame(), y.shape)
>>>>>>> origin/main
y = y.to_frame()
x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
<<<<<<< HEAD
layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSmlp", hp=True)
=======
layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.01}, x, y, splits, "arSmlp")
>>>>>>> origin/main
argrid.to_csv("./NmlpAS.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(800)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpmlp", hp=True)

print( 'hyperparameters: ', hpars)
grid.to_csv("./NmlpHP.csv")

