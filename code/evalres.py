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
import temb


x, y, xt = utils.loaddata('validation', 1, dir="./data/", raw=True)


yp_tr = pd.read_csv("./data/train_hyt.csv")
yp_te = pd.read_csv("./data/test_hyt.csv")
yp_tr.index = pd.DatetimeIndex(yp_tr['date'])
yp_te.index = pd.DatetimeIndex(yp_te['date'])

yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
ypte = yp_te.drop(yp_te.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)

#yp = yptr.merge(ypte, how="outer")

#print(len(yptr), len(ypte))
#print(yptr, ypte)
#yp = pd.concat([yptr, ypte])
#print(yp)


n = [1,1]
x_tr, n = utils.add_history(yptr, n, 1)
x_te, n = utils.add_history(ypte, n, 1)
x_tr = utils.standardize(x_tr)
x_te = utils.standardize(x_te)

y = y.to_frame()
train_x = x_tr[~x_tr.index.year.isin([2007,2008])]
train_y = y[~y.index.year.isin([2007,2008])]
splits = len(train_x.index.year.unique())

test_x = x_te[x_te.index.year == 2008]
test_y = y[y.index.year == 2008][1:]

print(train_x, train_y)


#print(len(x), len(y))
splits = len(train_x.index.year.unique())
#print(splits)

train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))

print("train_x", train_x)

# Load results from NAS
# Architecture
res_as = pd.read_csv("NresAS.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
layersizes = list(b[np.isfinite(b)].astype(np.int))

print('layersizes', layersizes)

model_design = {'layersizes': layersizes}

res_hp = pd.read_csv("NresHP.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
lr = b[0]
bs = b[1]


hp = {'epochs': 5000,
      'batchsize': int(bs),
      'lr': lr
      }

data_dir = "./data/"
data = "res"

tloss = temb.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, res=1, ypreles=None)
#pd.DataFrame.from_dict(tloss).to_csv('res_train.csv')

# Evaluation
mse = nn.MSELoss()
mae = nn.L1Loss()
x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)


train_rmse = []
train_mae = []
test_rmse = []
test_mae = []

preds_tr = {}
preds_te = {}
for i in range(splits):
    i += 1
    #import model
    model = models.NMLP(x_train.shape[1], y.shape[1], model_design['layersizes'])
    model.load_state_dict(torch.load(''.join((data_dir, f"res_model{i}.pth"))))
    model.eval()
    with torch.no_grad():
        p_train = model(x_train)
        p_test = model(x_test)
        preds_tr.update({f'train_res{i}':  p_train.flatten().numpy()})
        preds_te.update({f'test_res{i}':  p_test.flatten().numpy()})
        train_rmse.append(mse(p_train, y_train).tolist())
        train_mae.append(mae(p_train, y_train).tolist())
        test_rmse.append(mse(p_test, y_test).tolist())
        test_mae.append(mae(p_test, y_test).tolist())


performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}




pd.DataFrame.from_dict(performance).to_csv('res_eval_performance.csv')
pd.DataFrame.from_dict(preds_tr).to_csv('res_eval_preds_train.csv')
pd.DataFrame.from_dict(preds_te).to_csv('res_eval_preds_test.csv')

