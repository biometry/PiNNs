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

def ENres2(data_use='full'):
    print(os.getcwd())
    if data_use == 'sparse':
        x, y, xt = utils.loaddata('NAS', 1, dir="../../data/", raw=True, sparse=True)
    else:
        x, y, xt = utils.loaddata('NAS', 1, dir="../../data/", raw=True)
    ypreles = xt.drop(xt.columns.difference(['GPPp']), axis=1)[1:]

    splits = len(x.index.year.unique())

    y = y.to_frame()
    x.index, y.index, ypreles.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(ypreles))
            
    arch_grid, par_grid = HP.NASSearchSpace(x.shape[1], y.shape[1], 300, 300, 4)
    res = HP.NASSearch(arch_grid, par_grid, x, y, splits, "NASpres2", res=2, ypreles=ypreles, hp=True)
    res.to_csv(f"/results/Nres2HP_{data_use}.csv")


if __name__ == '__main__':
    ENres2(args.d)


