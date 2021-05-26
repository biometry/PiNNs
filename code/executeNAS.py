# !/usr/bin/env python
# coding: utf-8
import utils
import NAS


# import data
x, y = utils.loaddata('NAS', 1)

# architecture grid
arch_grid = NAS.ArchitectureSearchSpace(x.shape[1], y.shape[1], 100, 5)

# architecture search
layersizes = NAS.ArchitectureSearch(arch_grid, parameters={'epochs': 300, 'batchsize': 8, 'learningrate':0.01})

# Hyperparameter Search Space
hpar_grid = NAS.HParSearchSpace(100)

# Hyperparameter search
hpars = NAS.HParSearch(layersizes, hpar_grid)

print('Executed NAS \n layersizes: ', layersizes, '\n hyperparameters: ', hpars)

