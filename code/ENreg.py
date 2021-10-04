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


yp = xt.drop(xt.columns.difference(['GPPp']), axis=1)
reg = yp[1:]
y = y.to_frame()

splits = len(x.index.year.unique())

print(x,y, reg)

x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))


#arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
#layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001, "eta": 0.2}, x, y, splits, "arSreg", reg, hp=True)
#ag.to_csv("./NregAS.csv")
layersizes = [2, 128, 128, 128]

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(500, True)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpreg", reg, hp=True)

print( 'hyperparameters: ', hpars)


grid.to_csv("./NregHP.csv")

