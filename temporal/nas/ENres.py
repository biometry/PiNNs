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


def ENres(data_use="full"):
    
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('NASp', 0, dir="../../data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('NASp', 0, dir="../../data/", raw=True)
    y = y.to_frame()
    
    splits = len(x.index.year.unique())
    
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))
    
    arch_grid, par_grid = HP.NASSearchSpace(x.shape[1], y.shape[1], 300, 300, 4)
    res = HP.NASSearch(arch_grid, par_grid, x, y, splits, "NASres", hp=True)
    res.to_csv(f"/results/NresHP_{data_use}.csv")
if __name__ == '__main__':
    ENres(args.d)

