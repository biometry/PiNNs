# !/usr/bin/env python
# coding: utf-8
# @author: Niklas Moser
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from misc import HP
from misc import utils
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()


def EN2res2(data_use='full'):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="../../data/", raw=True)

    yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
    yp.index = pd.DatetimeIndex(x.index)
    yp = yp[yp.index.year == 2004]
    x = x[x.index.year == 2004]
    y = y[y.index.year == 2004]
    y = y.to_frame()
    splits = 5
    x.index, y.index, yp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(yp))
    
    arch_grid, par_grid = HP.NASSearchSpace(x.shape[1], y.shape[1], 300, 300, 4)
    res = HP.NASSearch(arch_grid, par_grid, x, y, splits, "2hpres2",exp=2, ypreles=yp, hp=True)

    res.to_csv(f"./results/N2res2HP_{data_use}.csv")
if __name__ == '__main__':
    EN2res2(args.d)
