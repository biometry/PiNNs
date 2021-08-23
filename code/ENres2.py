# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import trainloaded
import temb
import torch
import pandas as pd
import numpy as np

x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)

ypreles = xt.drop(xt.columns.difference(['GPPp']), axis=1)[1:]

splits = len(x.index.year.unique())

y = y.to_frame()
x.index, y.index, ypreles.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypreles))

print("x",x,"y",y, ypreles)
arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
layersizes, agrid = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSres2", res=2, ypreles=ypreles, hp=True)
agrid.to_csv("./NresAS2.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(800)

# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpres2", res=2, ypreles=ypreles, hp=True)

print( 'hyperparameters: ', hpars)


grid.to_csv("./NresHP2.csv")


