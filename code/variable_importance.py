# !/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import numpy as np
import utils
import models
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import random
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv
import training
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
parser.add_argument('-m', metavar='model', type=str, help='define model: mlp, res, res2, reg, emb, da')
args = parser.parse_args()

def predict(test_x, test_y, m, data_use, yp=None):
    # Architecture
    res_as = pd.read_csv(f"./results/N{m}AS_{data_use}.csv")
    a = res_as.loc[res_as.ind_mini.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))

    model_design = {'layersizes': layersizes}
    data_dir = "./data/"
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    if m == 'res2':
        yp_test = torch.tensor(yp.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    test_rmse = []
    test_mae = []

    preds_test = {}

    for i in range(4):
        i += 1
        #import model
        if m in ['mlp', 'res', 'reg']:
            model = models.NMLP(x_test.shape[1], 1, model_design['layersizes'])
        elif m == 'res2':
            model = models.RES(x_test.shape[1], 1, model_design['layersizes'])
        

        model.load_state_dict(torch.load(''.join((data_dir, f"{m}_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            if m == 'res2':
                p_test = model(x_test, yp_test)
            else:
                p_test = model(x_test)
            preds_test.update({f'test_{m}{i}': p_test.flatten().numpy()})



    return preds_test




def via(data_use, model, yp=None):

    if data_use == 'sparse':
        x, y, xt, mn, std = utils.loaddata('validation', 1, dir="./data/", raw=True, sparse=True, via=True)
        if model in ['res', 'res2']:
            yp = utils.make_sparse(pd.read_csv("./data/Hyytiala.csv"))
    else:
        x, y, xt, mn, std = utils.loaddata('validation', 1, dir="./data/", raw=True, via=True)
        if model in ['res', 'res2']:
            yp = pd.read_csv("./data/Hyytiala.csv")   

    
    thresholds = {'PAR': [0, 200], 
                  'Tair': [-20, 40],
                  'VPD': [0, 60],
                  'Precip': [0, 100],
                  #'co2': [],
                  'fapar': [0, 1],
                  'GPPp': [0, 30],
                  'ETp': [0, 800],
                  'SWp': [0, 400]
    }

    gridsize = 200
    
    if model == 'res2':
        yp.index = pd.DatetimeIndex(yp['date'])
        yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
        yp = yp[yp.index.year == 2008][1:]
        
    if model == 'res':
        yp.index = pd.DatetimeIndex(yp['date'])
        yp = yp.drop(yp.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        n = [1,1]
        x_te, n = utils.add_history(yp, n, 1)
        x_te, mn, std = utils.standardize(x_te, get_p=True)
        test_x = x_te[x_te.index.year == 2008]
        test_y = y[y.index.year == 2008]
        variables = ['GPPp', 'ETp', 'SWp']
                
    elif model in ['mlp', 'res2', 'reg']:
        test_x = x[x.index.year == 2008]
        test_y = y[y.index.year == 2008]
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']
    dates = test_x.index.copy()
    print(test_x, test_y, yp)

    for v in variables:
        if model == 'res':
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[''.join((v, '_x'))])/std[''.join((v, '_x'))]
        else:
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[v])/std[v] 
        output_cols = np.linspace(thresholds[v][0], thresholds[v][1], gridsize)
        output = pd.DataFrame()
        output['date'] = dates

        for i in range(gridsize):
            test_x[''.join((v, '_x'))] = [var_range[i]]*len(dates)
            test_x[''.join((v, '_y'))] = [var_range[i]]*len(dates)
            test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))

            ps = predict(test_x, test_y, model, data_use, yp)

            output[output_cols[i]] = pd.DataFrame.from_dict(ps).apply(lambda row: np.mean(row.to_numpy()), axis=1)

        output.T.to_csv(f'./results/{model}_{data_use}_{v}_via_marginal.csv')


if __name__ == '__main__':
    via('full', 'mlp')
    #via('sparse', 'res2')
    #via('sparse', 'res')
