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

x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)


# select NAS data
print(x.index)
x = x[x.index.year == 2004]
y = y[y.index.year == 2004]
x = x.drop(pd.DatetimeIndex(['2004-01-01']))
y = y.drop(pd.DatetimeIndex(['2004-01-01']))
print(x,y)

splits = 8
print(splits)
print(x, y)
y = y.to_frame()
x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))


arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001}, x, y, splits, "EX2_arSmlp", exp=2, hp=True)
argrid.to_csv("./EX2_mlpAS.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(800)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "EX2_hpmlp", exp=2, hp=True)

print( 'hyperparameters: ', hpars)

grid.to_csv("./EX2_mlpHP.csv")

