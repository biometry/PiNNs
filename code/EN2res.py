# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import torch
import pandas as pd
import numpy as np
import argparse
import HPe
parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

def EN2res(data_use='full', v=2):
    print('data')
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('exp2p', 0, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('exp2p', 0, dir="./data/", raw=True)

        x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        y = y.drop(pd.DatetimeIndex(['2004-01-01']))

    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]
    y = y.to_frame()
    print('XY',x, y)
    #print(len(x), len(y))
    splits = 5
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))
    if v!=2:
        arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)
        # architecture search
        layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001}, x, y, splits, "EX2_arSres", exp=2, hp=True)
        ag.to_csv(f"/scratch/project_2000527/pgnn/results/EX2_resAS_{data_use}.csv")
        #layersizes = [4, 32, 2, 16]
        # Hyperparameter Search Space
        hpar_grid = HP.HParSearchSpace(800)
        # Hyperparameter search
        hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "EX2_hpres", exp=2, hp=True)
        print( 'hyperparameters: ', hpars)
        grid.to_csv(f"/scratch/project_2000527/pgnn/results/EX2_resHP_{data_use}.csv")
    if v==2:
        arch_grid, par_grid = HPe.NASSearchSpace(x.shape[1], y.shape[1], 200, 200, 4)
        res = HPe.NASSearch(arch_grid, par_grid, x, y, splits, "2hpEMBres", exp=2, hp=True)

        res.to_csv(f"/scratch/project_2000527/pgnn/results/EX2EMBresHP_{data_use}_new.csv")

if __name__ == '__main__':
    EN2res(args.d)
