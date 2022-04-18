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


def predict(test_x, test_y):
    
    # Architecture
    res_as = pd.read_csv(f"results/NmlpAS_{data_use}.csv")
    a = res_as.loc[res_as.val_loss.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))

    model_design = {'layersizes': layersizes}
    data_dir = "./data/"
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    test_rmse = []
    test_mae = []

    preds_test = {}

    for i in range(4):
        i += 1
        #import model
        model = models.NMLP(x_test.shape[1], 1, model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"mlp_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_test = model(x_test)
            preds_test.update({f'test_mlp{i}': p_test.flatten().numpy()})



    return preds_test




data_use='full'

if data_use == 'sparse':
    x, y, xt, mn, std = utils.loaddata('validation', 1, dir="./data/", raw=True, sparse=True, via=True)
else:
    x, y, xt, mn, std = utils.loaddata('validation', 1, dir="./data/", raw=True, via=True)


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

test_x = x[x.index.year == 2008][1:]
test_y = y[y.index.year == 2008][1:]

t = test_x.index
var = 'Tair'
var_range = (np.linspace(thresholds[var][0], thresholds[var][1], gridsize)-mn[var])/std[var] 
dates = test_x.index

output_cols = np.linspace(thresholds[var][0], thresholds[var][1], gridsize)
output = pd.DataFrame()
output['date'] = dates

for i in range(gridsize):
    test_x[''.join((var, '_x'))] = [var_range[i]]*len(dates)
    test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
    
    ps = predict(test_x, test_y)

    output[output_cols[i]] = pd.DataFrame.from_dict(ps).apply(lambda row: np.mean(row.to_numpy()), axis=1)

print(output.T)



