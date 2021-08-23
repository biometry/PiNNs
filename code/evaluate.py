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


# Load hyytiala
x, y, mn, std, xt = utils.loaddata('validation', 1, dir="./data/", raw=True)

train_x = x[x.index.year != 2008]
train_y = y[y.index.year != 2008]
splits = len(train_x.index.year.unique())

test_x = x[x.index.year == 2008]
test_y = y[y.index.year == 2008]
train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))


# Load results from NAS
# Architecture
res_as = pd.read_csv("NmlpAS.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy(dtype=np.int)
layersizes = list(b[np.isfinite(b)])
print('layersizes', layersizes)

model_design = {'layersizes': layersizes}

res_hp = pd.read_csv("NmlpHP.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
lr = b[0]
bs = b[1]

hp = {'epochs': 5000,
      'batchsize': int(bs),
      'lr': lr}

data_dir = "./data/"
data = "mlp"
tloss = temb.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, mn=None, std=None)

# Evaluation
mse = nn.MSELoss()
mae = nn.L1Loss()
x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
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
    model.load_state_dict(torch.load(''.join((data_dir, f"mlp_model{i}.pth"))))
    model.eval()
    with torch.no_grad():
        p_train = model(x_train)
        p_test = model(x_test)
        preds_train.update({f'train_mlp{i}': p_train.flatten().numpy()*std['GPP']+mn['GPP']})
        preds_test.update({f'test_mlp{i}': p_test.flatten().numpy()*std['GPP']+mn['GPP']})
        train_rmse.append(mse(p_train, y_train).tolist())
        train_mae.append(mae(p_train, y_train).tolist())
        test_rmse.append(mse(p_test, y_test).tolist())
        test_mae.append(mae(p_test, y_test).tolist())

performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}


print(preds_train)


pd.DataFrame.from_dict(performance).to_csv('mlp_eval_performance.csv')
pd.DataFrame.from_dict(preds_train).to_csv('mlp_eval_preds_train.csv')
pd.DataFrame.from_dict(preds_test).to_csv('mlp_eval_preds_test.csv')



'''
with open('mlp_eval_performance.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, performance.keys())
    w.writeheader()
    w.writerow(performance)


with open('mlp_eval_preds.csv', 'w') as f:  # You will need 'wb' mode in Python 2.x
    w = csv.DictWriter(f, preds.keys())
    w.writeheader()
    w.writerow(preds)
'''
    



    















