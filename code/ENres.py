# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import utils
import torch
import pandas as pd
import numpy as np
import argparse
import HPe

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()


def ENres(data_use="full", v=2):
    
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('NASp', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('NASp', 1, dir="./data/", raw=True)
    y = y.to_frame()
 
            
    print("X",x, "Y",y)
    print('Length' ,len(x), len(y))
    
    splits = len(x.index.year.unique())
    
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))
    
    if v==1:
        # orignial grid size: 800
        arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 800, 4)
    
        # architecture search
        layersizes, agrid = HP.ArchitectureSearch(arch_grid, {'epochs': 200, 'batchsize': 8, 'lr':0.001}, x, y, splits, "arSres", hp=True)
    
        agrid.to_csv(f"/scratch/project_2000527/pgnn/results/NresAS_{data_use}.csv")
    
        # Hyperparameter Search Space
        # original search space size: 800
        hpar_grid = HP.HParSearchSpace(800)
    
        # Hyperparameter search
        hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpres", hp=True)
    
        print( 'hyperparameters: ', hpars)
        grid.to_csv(f"/scratch/project_2000527/pgnn/results/NresHP_{data_use}.csv")

    elif v==2:
        arch_grid, par_grid = HPe.NASSearchSpace(x.shape[1], y.shape[1], 100, 100, 4)
        res = HPe.NASSearch(arch_grid, par_grid, x, y, splits, "hpmlp", hp=True)
        
if __name__ == '__main__':
    ENres(args.d)

