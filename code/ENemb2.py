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
parser.add_argument('-o', metavar='optim', type=str, help='lbfgs?')
args = parser.parse_args()


def ENemb(data_use='full', opt='lbfgs', splits=None):
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
    y = y.to_frame()
    yp = xt.GPPp.to_frame()
    swmn = np.mean(xt.SWp)
    swstd = np.std(xt.SWp)
    xt = xt.drop(['date', 'ET', 'GPP', 'SWp', 'GPPp', 'ETp'], axis=1)
    
    if splits is None:
        splits = len(x.index.year.unique())
    
    x.index, y.index, yp.index, xt.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(yp)), np.arange(0, len(xt))
    
    arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 200, 4, emb=True)
    if opt == 'lbfgs':
        qn = True
    else:
        qn = False
    # architecture search
    
    print("DATA", x,y,xt,yp)
    layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 365, 'lr':0.001, 'eta': 0.2}, x, y, splits, "arSemb2", reg=yp, emb=True, raw = xt, embtp=2, hp=True, sw=(swmn, swstd), qn = qn)
    ag.to_csv(f"./Nemb2_{data_use}_AS.csv")

    
    # Hyperparameter Search Space
    #hpar_grid = HP.HParSearchSpace(1, reg=True, emb=True)

    # Hyperparameter search
    #hpars, grid = HP.HParSearch(layersizes, hpar_grid, x, y, splits, "hpemb2", reg=yp, emb=True, raw=xt, embtp=2, hp=True, sw=(swmn, swstd), qn =qn)

    #print( 'hyperparameters: ', hpars)
    #grid.to_csv("./Nemb2HP_m300.csv")


if __name__=='__main__':
    ENemb(args.d, args.o)
