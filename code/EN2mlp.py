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
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

def EN2mlp(data_use='full', splits=None):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)
        x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        y = y.drop(pd.DatetimeIndex(['2004-01-01']))
    # select NAS data
    print(x.index)
    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]

    print(x,y)
    
    
    
    splits = 5
    print(splits)
    print(x, y)
    y = y.to_frame()
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))


    arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)
    
    # architecture search
    layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001}, x, y, splits, "EX2_arSmlp", exp=2, hp=True)
    argrid.to_csv(f"/scratch/project_2000527/pgnn/results/EX2_mlpAS_{data_use}.csv")

    # Hyperparameter Search Space
    hpar_grid = HP.HParSearchSpace(800)

    # Hyperparameter search
    hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "EX2_hpmlp", exp=2, hp=True)

    print( 'hyperparameters: ', hpars)

    grid.to_csv(f"/scratch/project_2000527/pgnn/results/EX2_mlpHP_{data_use}.csv")


if __name__ == '__main__':
    EN2mlp(args.d)
