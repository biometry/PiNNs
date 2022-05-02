# !/usr/bin/env python
# coding: utf-8
import utils
import torch
import pandas as pd
import numpy as np
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
args = parser.parse_args()


def eval2mlp(data_use='full'):
    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)



    # select NAS data
    print(x.index)
    x = x[x.index.year==2005]
    y = y[y.index.year==2005]
    x = x.drop(pd.DatetimeIndex(['2005-01-01']))
    y = y.drop(pd.DatetimeIndex(['2005-01-01']))

    test_x = x[x.index.year==2008]
    test_y = y[y.index.year==2008]
    #test_x = test_x.drop(pd.DatetimeIndex(['2008-01-01']))
    #test_y = test_y.drop(pd.DatetimeIndex(['2008-01-01']))    

    splits = 5
    print(splits)
    print(test_x, test_y)
    y = y.to_frame()
    x.index, y.index = np.arange(0, len(x)), np.arange(0, len(y))

    mlp_as = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_mlpAS_{data_use}.csv")
    a = mlp_as.loc[mlp_as.val_loss.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))
    print('layersizes', layersizes)
    
    model_design = {'layersizes': layersizes}

    
    mlp_hp = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_mlpHP_{data_use}.csv")
    a = mlp_hp.loc[mlp_hp.val_loss.idxmin()][1:3]
    b = a.to_numpy()
    lr = b[0]
    bs = b[1]
    
    hp = {'epochs': 5000,
      'batchsize': int(bs),
            'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "./data/"
    data = f"mlp_{data_use}"
    #print('DATA', train_x, train_y)
    #print('TX', train_x, train_y)


    td, se, ae = training.train_cv(hp, model_design, x, y, data_dir, splits, data, reg=None, emb=False, exp=2)
    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_{data_use}_trainloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_{data_use}_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_{data_use}_vaeloss.csv')
    
    
     # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train = torch.tensor(x.to_numpy(), dtype=torch.float32), torch.tensor(y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    
    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models.NMLP(x.shape[1], y.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"2mlp_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train)
            p_test = model(x_test)
            preds_train.update({f'train_mlp{i}': p_train.flatten().numpy()})
            preds_test.update({f'test_mlp{i}': p_test.flatten().numpy()})
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())

    performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}


    print(preds_train)



    pd.DataFrame.from_dict(performance).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_train).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_{data_use}_eval_preds_train.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_{data_use}_eval_preds_test.csv')


    

if __name__ == '__main__':
    eval2mlp(args.d)
