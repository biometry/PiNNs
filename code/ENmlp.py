# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import torch
import pandas as pd
import numpy as np

x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
print(x,y,xt)
splits = len(x.index.year.unique())
print(splits)
print(x, y)

y = y.to_frame()
x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 5, 4)

# architecture search
# original: use grid of 800 and epochs:100
layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 10, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSmlp", hp=True)
argrid.to_csv("./results/NmlpAS.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(5)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpmlp", hp=True)

print( 'hyperparameters: ', hpars)
grid.to_csv("./results/NmlpHP.csv")

