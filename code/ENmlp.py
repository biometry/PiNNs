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

print(args)

def ENmlp(data_use="full", splits=None):

    x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
    y = y.to_frame()

    if data_use == "sparse":
        x, y = utils.sparse(x, y)
    
    if splits is None:
        splits = len(x.index.year.unique())
    
    print(splits)
    print(x, y)
    
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

    arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 5, 4)

    # architecture search
    # original: use grid of 800 and epochs:100
    layersizes, argrid = HP.ArchitectureSearch(arch_grid, {'epochs': 10, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSmlp", hp=True)
    argrid.to_csv(f"~/physics_guided_nn/results/NmlpAS_{data_use}.csv")

    # Hyperparameter Search Space
    hpar_grid = HP.HParSearchSpace(5)
    
    # Hyperparameter search
    hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y,  splits, "hpmlp", hp=True)
    
    print( 'hyperparameters: ', hpars)
    grid.to_csv(f"./results/NmlpHP_{data_use}.csv")


if __name__ == '__main__':
    ENmlp(args.d)
