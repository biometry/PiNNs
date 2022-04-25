# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def EN2reg(data_use='full'):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)
        #x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        #y = y.drop(pd.DatetimeIndex(['2004-01-01']))
        #yp = yp.drop(pd.DatetimeIndex(['2004-01-01']))
    yp.index = x.index
    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]
    yp = yp[yp.index.year == 2004]


    reg = yp.GPPp.to_frame()
    y = y.to_frame()

    #print(len(x), len(y))
    splits = 5
    #print(splits)
    print(x,y, reg)

    x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))

    arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

    # architecture search
    layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001, "eta": 0.2}, x, y, splits, "EX2_arSreg", reg, exp=2, hp=True)
    ag.to_csv(f"./results/EX2_regAS_{data_use}.csv")

    #layersizes = [4, 32, 2, 16]
    # Hyperparameter Search Space
    hpar_grid = HP.HParSearchSpace(800, True)

    # Hyperparameter search
    hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "EX2_hpreg", reg, exp=2, hp=True)

    print( 'hyperparameters: ', hpars)
    grid.to_csv(f"./results/EX2_regHP_{data_use}.csv")

if __name__ == '__main__':
    EN2reg(args.d)

