# !/usr/bin/env python
# coding: utf-8
import utils
import NASemb


# import data
x, y, raw = utils.loaddata('NAS', 0, dir = './data/', raw=True)

xn = raw.drop(['date', 'GPP', 'ET'], axis=1)
print(xn)

# architecture grid
arch_grid = NASemb.ArchitectureSearchSpace(x.shape[1], y.shape[1], 200, 7)


# architecture search
layersizes = NASemb.ArchitectureSearch(arch_grid, parameters={'epochs': 300, 'batchsize': 8, 'learningrate':0.01}, X=x, Y=y, Xn=xn)

# Hyperparameter Search Space
hpar_grid = NASemb.HParSearchSpace(100)

# Hyperparameter search
hpars = NASemb.HParSearch(layersizes, hpar_grid, x, y, Xn=xn)

print('Executed NAS \n layersizes: ', layersizes, '\n hyperparameters: ', hpars)
