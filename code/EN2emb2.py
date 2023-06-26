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
parser.add_argument('-o', metavar='optim', type=str, help='lbfgs?')
args = parser.parse_args()


def ENemb(data_use='full', opt='lbfgs', splits=None):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 0, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 0, dir="./data/", raw=True)
    y = y.to_frame()
    swmn = np.mean(yp.SWp)
    swstd = np.std(yp.SWp)
    xt = xt.drop(['Unnamed: 0', 'Unnamed: 0.1', 'X.1','X','date', 'ET', 'GPP', 'SWp', 'GPPp', 'ETp', 'year', 'site'], axis=1)
    yp.index = y.index
    yp = yp[yp.index.year == 2004]
    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]
    yp = yp.GPPp.to_frame()
    print('xyypt',xt, y, yp)
    splits = 8
    
    x.index, y.index, yp.index, xt.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(yp)), np.arange(0, len(xt))
    
    arch_grid = HP.ArchitectureSearchSpace(x.shape[1], y.shape[1], 300, 4, emb=True)
    if opt == 'lbfgs':
        qn = True
    else:
        qn = False
    # architecture search
    
    print("DATA", type(x),type(y),type(xt),type(yp))
    layersizes, ag = HP.ArchitectureSearch(arch_grid, {'epochs': 100, 'batchsize': 365, 'lr':0.001, 'eta': 0.2}, x, y, splits, "arSemb2", reg=yp, emb=True, raw = xt, embtp=2, hp=True, sw=(swmn, swstd), qn = qn, exp=2)
    ag.to_csv(f"./EX2Nemb2_{data_use}_AS.csv")

if __name__=='__main__':
    ENemb(args.d, args.o)





