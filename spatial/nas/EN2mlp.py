# !/usr/bin/env python
# coding: utf-8
# @author: Niklas Moser
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from misc import HP
from misc import utils
from misc import models
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def EN2mlp(data_use='full', splits=None):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True)
        x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        y = y.drop(pd.DatetimeIndex(['2004-01-01']))
    # select NAS data
    print(x.index.year)
    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]
    
    splits = 5
    y = y.to_frame()
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

    arch_grid, par_grid = HP.NASSearchSpace(x.shape[1], y.shape[1], 300, 300, 4)
    res = HP.NASSearch(arch_grid, par_grid, x, y, splits, "2hpmlp", exp=2, hp=True)

    res.to_csv(f"./results/N2mlpHP_{data_use}.csv")

if __name__ == '__main__':
    EN2mlp(args.d)
