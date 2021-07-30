# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import trainloaded
import embtraining
import torch
import pandas as pd

x, y, mn, std, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
ypreles = pd.read_csv('./data/soro_p.csv')
yp = ypreles['GPPp']


print(x,y,mn,std,xt)

xn = xt.drop(['date', 'GPP', 'ET'], axis=1)
yy = y.drop(['ET'], axis=1)

#model_design = {'layer_sizes': [[16, 16], [8, 8]]}


# architecture grid
print(x.shape, y.shape)
arch_grid = HP.ArchitectureSearchSpace(50, 3)
print(arch_grid)
# architecture search
layersizes = HP.ArchitectureSearch(arch_grid, {'epochs': 300, 'batchsize': 2, 'learningrate':0.01}, x, yy, xn, mn, std, yp)

# Hyperparameter Search Space
hpar_grid = HP.HParSearchSpace(100)
#layersizes = [[16,16], [8, 8]]
# Hyperparameter search
hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, yy, xn, mn, std, yp)

print( 'hyperparameters: ', hpars)


grid.to_csv("./outEMBNAS.csv")
