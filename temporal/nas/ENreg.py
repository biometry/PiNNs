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

def ENreg(data_use='full', splits=None):
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('NAS', 1, dir="../../data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('NAS', 1, dir="../../data/", raw=True)
    yp = xt.drop(xt.columns.difference(['GPPp']), axis=1)
    reg = yp[1:]
    y = y.to_frame()

    if splits == None:
        splits = len(x.index.year.unique())

    print("INPUTS: \n", x, "Outputs: \n", y, "RAW DATA: \n", reg)
    x.index, y.index, reg.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(reg))

    arch_grid, par_grid = HP.NASSearchSpace(x.shape[1], y.shape[1], 300, 300, 4, reg=True)
    res = HP.NASSearch(arch_grid, par_grid, x, y, splits, "NASreg", reg=reg, hp=True)
    res.to_csv(f"/results/NregHP_{data_use}.csv")

if __name__ == '__main__':
    ENreg(args.d)

