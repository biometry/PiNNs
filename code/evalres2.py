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
yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp']), axis=1)
ypte = yp_te.drop(yp_te.columns.difference(['GPPp']), axis=1)
#yp = yptr.merge(ypte, how="outer")

#print(len(yptr), len(ypte))
#print(yptr, ypte)
#yp = pd.concat([yptr, ypte])
#print(yp)


yp_tr = yptr[~yptr.index.year.isin([2007])][1:]
yp_te = ypte[1:]
y = y.to_frame()
train_x = x[~x.index.year.isin([2007,2008])]
train_y = y[~y.index.year.isin([2007,2008])]
splits = len(train_x.index.year.unique())
print(train_x, train_y)
print(yp_tr)


test_x = x[x.index.year == 2008]
test_y = y[y.index.year == 2008]
print(test_x)


#print(len(x), len(y))
splits = len(train_x.index.year.unique())
#print(splits)
train_x.index, train_y.index, yp_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr)) 
test_x.index, test_y.index, yp_te.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(yp_te))
print("train_x", train_x)

# Load results from NAS
# Architecture
res_as = pd.read_csv("NresAS2.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
layersizes = list(b[np.isfinite(b)].astype(np.int))

print('layersizes', layersizes)

model_design = {'layersizes': layersizes}

res_hp = pd.read_csv("NresHP2.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
lr = b[0]
bs = b[1]

hp = {'epochs': 5000,
      'batchsize': int(bs),
      'lr': lr
      }

print(hp)

data_dir = "./data/"
data = "res2"
tloss = temb.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, res=2, ypreles=yp_tr, exp=1)
#pd.DataFrame.from_dict(tloss).to_csv('res2_test.csv')

# Evaluation
mse = nn.MSELoss()
mae = nn.L1Loss()
x_train, y_train, tr_yp = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(yp_tr.to_numpy(), dtype=torch.float32)

x_test, y_test, te_yp = torch.tensor(test_x.to_numpy()[1:], dtype=torch.float32), torch.tensor(test_y.to_numpy()[1:], dtype=torch.float32), torch.tensor(yp_te.to_numpy(), dtype=torch.float32)

train_rmse = []
train_mae = []
test_rmse = []
test_mae = []

preds_tr = {}
preds_te = {}
for i in range(splits):
    i += 1
    #import model
    model = models.RES(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
    model.load_state_dict(torch.load(''.join((data_dir, f"res2_model{i}.pth"))))
    model.eval()
    with torch.no_grad():
        p_train = model(x_train, tr_yp)
        p_test = model(x_test, te_yp)

        preds_tr.update({f'train_res2{i}':  p_train.flatten().numpy()})
        preds_te.update({f'test_res2{i}':  p_test.flatten().numpy()})

        train_rmse.append(mse(p_train, y_train).tolist())
        train_mae.append(mae(p_train, y_train).tolist())
        test_rmse.append(mse(p_test, y_test).tolist())
        test_mae.append(mae(p_test, y_test).tolist())


performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}


pd.DataFrame.from_dict(performance).to_csv('res2_eval_performance.csv')
pd.DataFrame.from_dict(preds_tr).to_csv('res2_eval_preds_train.csv')
pd.DataFrame.from_dict(preds_te).to_csv('res2_eval_preds_test.csv')
