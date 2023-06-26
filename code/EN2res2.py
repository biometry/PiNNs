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
args = parser.parse_args()


def EN2res2(data_use='full', v=2):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)
        #x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        #y = y.drop(pd.DatetimeIndex(['2004-01-01']))
        #yp = yp.drop(pd.DatetimeIndex(['2004-01-01']))

    yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
    yp.index = pd.DatetimeIndex(x.index)
    yp = yp[yp.index.year == 2004]
    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]
    y = y.to_frame()
    print('XYYP',x, y, yp)
    #print(len(x), len(y))
    splits = 5
    x.index, y.index, yp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(yp))
    if v!=1:
        arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)
        # architecture search
        layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001}, x, y, splits, "EX2_arSres2", exp=2, hp=True, res=2, ypreles = yp)
        ag.to_csv(f"/scratch/project_2000527/pgnn/results/EX2_res2AS_{data_use}.csv")
        #layersizes = [4, 32, 2, 16]
        # Hyperparameter Search Space
        hpar_grid = HP.HParSearchSpace(800)
        # Hyperparameter search
        hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "EX2_hpres2", exp=2, hp=True, res=2, ypreles = yp)
        print( 'hyperparameters: ', hpars)
        grid.to_csv(f"/scratch/project_2000527/pgnn/results/EX2_res2HP_{data_use}.csv")
    if v==2:
        arch_grid, par_grid = HPe.NASSearchSpace(x.shape[1], y.shape[1], 130, 130, 4)
        res = HPe.NASSearch(arch_grid, par_grid, x, y, splits, "2hpres2",exp=2, ypreles=yp, hp=True)

        res.to_csv(f"/scratch/project_2000527/pgnn/results/EX2Nres2HP_{data_use}_new.csv")
if __name__ == '__main__':
    EN2res2(args.d)
