# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import training
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

def ENreg(data_use='full', splits=2):

    x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
    #print("INPUTS: \n", x, "Outputs: \n", y, "RAW DATA: \n", reg)
    yp = xt.drop(xt.columns.difference(['GPPp']), axis=1)

    reg = yp[1:]
    y = y.to_frame()

    if data_use == 'sparse':
        x, y, reg = utils.sparse(x, y, reg)
    if splits == None:
        splits = len(x.index.year.unique())

    print("Use splits: ", splits)
    print("INPUTS: \n", x, "Outputs: \n", y, "RAW DATA: \n", reg)
    x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))


    arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 500, 4)

    # architecture search
    layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 8, 'lr':0.001, "eta": 0.2}, x, y, splits, "arSreg", reg, hp=True)
    ag.to_csv(f"./results/NregAS_{data_use}.csv")

    # Hyperparameter Search Space
    # originally: 500
    hpar_grid = HP.HParSearchSpace(500, True)

    # Hyperparameter search
    hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpreg", reg, hp=True)

    print( 'hyperparameters: ', hpars)
    grid.to_csv(f"./results/NregHP_{data_use}.csv")


if __name__ == '__main__':
    ENreg(args.d)

