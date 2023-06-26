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

def ENres2(data_use='full', v=2):
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
    ypreles = xt.drop(xt.columns.difference(['GPPp']), axis=1)[1:]

    splits = len(x.index.year.unique())

    y = y.to_frame()
    print("x----",x,"y",y, ypreles)
    x.index, y.index, ypreles.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypreles))
    if v==1:
        print("x----",x,"y",y, ypreles)
        arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)

        # architecture search
        layersizes, agrid = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSres2", res=2, ypreles=ypreles, hp=True)
        agrid.to_csv(f"/scratch/project_2000527/pgnn/results/Nres2AS_{data_use}.csv")

        # Hyperparameter Search Space
        hpar_grid = HP.HParSearchSpace(800)

        # Hyperparameter search
        hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpres2", res=2, ypreles=ypreles, hp=True)
        print( 'hyperparameters: ', hpars)
        grid.to_csv(f"/scratch/project_2000527/pgnn/results/Nres2HP_{data_use}.csv")
        
    elif v==2:
        arch_grid, par_grid = HPe.NASSearchSpace(x.shape[1], y.shape[1], 300, 300, 4)
        res = HPe.NASSearch(arch_grid, par_grid, x, y, splits, "NASpres2", res=2, ypreles=ypreles, hp=True)
        res.to_csv(f"/scratch/project_2000527/pgnn/results/Nres2HP_{data_use}_new.csv")


if __name__ == '__main__':
    ENres2(args.d)


