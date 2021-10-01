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

<<<<<<< HEAD
ypreles = xt.drop(xt.columns.difference(['GPPp']), axis=1)[1:]

splits = len(x.index.year.unique())

y = y.to_frame()
x.index, y.index, ypreles.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypreles))

print("x",x,"y",y, ypreles)
arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
layersizes, agrid = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSres2", res=2, ypreles=ypreles, hp=True)
=======
yp_tr = pd.read_csv("./data/train_soro.csv")
yp_te = pd.read_csv("./data/test_soro.csv")
yp_tr.index = pd.DatetimeIndex(yp_tr['date'])
yp_te.index = pd.DatetimeIndex(yp_te['date'])
yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp']), axis=1)
ypte = yp_te.drop(yp_te.columns.difference(['GPPp']), axis=1)
#yp = yptr.merge(ypte, how="outer")

#print(len(yptr), len(ypte))
#print(yptr, ypte)
yp = pd.concat([yptr, ypte])

ypreles = yp[1:]
#print(len(x), len(y))
splits = len(x.index.year.unique())
#print(splits)
y = y.to_frame()
x.index, y.index, ypreles.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypreles))

print("x",x,"y",y)
arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

# architecture search
layersizes, agrid = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.01}, x, y, splits, "arSres2", res=2, ypreles=ypreles)
>>>>>>> origin/main
agrid.to_csv("./NresAS2.csv")

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(800)

# Hyperparameter search
<<<<<<< HEAD
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpres2", res=2, ypreles=ypreles, hp=True)
=======
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpres2", res=2, ypreles=ypreles)
>>>>>>> origin/main

print( 'hyperparameters: ', hpars)


grid.to_csv("./NresHP2.csv")


