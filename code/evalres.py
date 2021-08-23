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

x, y, mn, std, xt = utils.loaddata('validation', 0, dir="./data/", raw=True)

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

yp_tr = (yptr-mn['GPP'])/std['GPP']
yp_te = (ypte-mn['GPP'])/std['GPP']

train_x = x[x.index.year != 2008]
train_y = y[y.index.year != 2008]
splits = len(train_x.index.year.unique())

test_x = x[x.index.year == 2008]
test_y = y[y.index.year == 2008]


#print(len(x), len(y))
splits = len(train_x.index.year.unique())
#print(splits)
train_x.index, train_y.index, yp_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr)) 
test_x.index, test_y.index, yp_te.index = np.arange(0, len(test_y)), np.arange(0, len(test_y)), np.arange(0, len(yp_te))
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
#tloss = temb.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, mn=None, std=None, res=1, ypreles=yp_tr)

# Evaluation
mse = nn.MSELoss()
mae = nn.L1Loss()
x_train, y_train = torch.tensor(yp_tr.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
x_test, y_test = torch.tensor(yp_te.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

train_rmse = []
train_mae = []
test_rmse = []
test_mae = []

preds_tr = {}
preds_te = {}
for i in range(splits):
    i += 1
    #import model
    model = models.NMLP(yp_tr.shape[1], y.shape[1], model_design['layersizes'])
    model.load_state_dict(torch.load(''.join((data_dir, f"res_model{i}.pth"))))
    model.eval()
    with torch.no_grad():
        p_train = model(x_train)
        p_test = model(x_test)
        preds_tr.update({f'train_reg{i}':  p_train.flatten().numpy()*std['GPP']+mn['GPP']})
        preds_te.update({f'test_reg{i}':  p_test.flatten().numpy()*std['GPP']+mn['GPP']})
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

