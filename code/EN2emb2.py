# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import torch
import pandas as pd
import numpy as np

x, y, xt, yp = utils.loaddata('exp2', 0, dir="./data/", raw=True)

swmn = np.mean(yp.SWp)
swstd = np.std(yp.SWp)
xt = xt.drop(['date', 'ET', 'GPP', 'X', 'Unnamed: 0', 'GPPp', 'ETp', 'SWp'], axis = 1)
yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)

yp = yp[yp.index.year == 2004]
x = x[x.index.year == 2004]
y = y[y.index.year == 2004]
y = y.to_frame()
print(x, y, yp)
splits = 8
x.index, y.index, yp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(yp))

#arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 140, 4, emb=True)

# architecture search
#layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001, 'eta':0.2}, x, y, splits, "EX2_arSemb2", exp=2, hp=True, emb=True, reg=yp, raw = xt, embtp=2, sw=(swmn, swstd))
#ag.to_csv("./EX2_emb2AS.csv")

layersizes = [[8, 8, 32], [4, 8, 8]]
# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(150, True, emb=True)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "EX2_hpemb2", exp=2, hp=True, emb=True, reg = yp, raw = xt, embtp=2, sw=(swmn, swstd))

print( 'hyperparameters: ', hpars)


grid.to_csv("./EX2_emb2HP_mn300.csv")

