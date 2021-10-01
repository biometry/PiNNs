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
import cv


# Load hyytiala
x, y, xt = utils.loaddata('validation', 1, dir="./data/", raw=True)
y = y.to_frame()


print(x.index.year.unique())
train_x = x[~x.index.year.isin([2007,2008])]
train_y = y[~y.index.year.isin([2007,2008])]

splits = len(train_x.index.year.unique())

test_x = x[x.index.year == 2008]
test_y = y[y.index.year == 2008]
train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
print('XY: ', x, y)

# Load results from NAS
# Architecture
res_as = pd.read_csv("NmlpAS.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
layersizes = list(b[np.isfinite(b)].astype(int))
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
print('HYPERPARAMETERS', hp)
data_dir = "./data/"
data = "mlp"
#print('DATA', train_x, train_y)
#print('TX', train_x, train_y)
tloss = temb.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False)
#tloss = temb.train(hp, model_design, train_x, train_y, data_dir, None, data, reg=None, emb=False)

#print("LOSS", tloss)
#pd.DataFrame(tloss, index=[0]).to_csv('mlp_train2.csv')


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
'''
model = models.NMLP(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
model.load_state_dict(torch.load(''.join((data_dir, 'modelev2.pth'))))
model.eval()
with torch.no_grad():
    train = model(x_train)
    test = model(x_test)
    print(mse(train, y_train))
    print(mae(test, y_test))
    print(train)
    print(test)

<<<<<<< HEAD

# Load results from NAS
# Architecture
res_as = pd.read_csv("NmlpAS.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
layersizes = list(b[np.isfinite(b)].astype(int))
print('layersizes', layersizes)

model_design = {'layersizes': layersizes}

res_hp = pd.read_csv("NmlpHP.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
bs = b[1]

res_hp = pd.read_csv("mlp_lr.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
lr = b[0]



hp = {'epochs': 5000,
      'batchsize': int(bs),
      'lr': lr}
print('HYPERPARAMETERS', hp)
data_dir = "./data/"
data = "mlp"
#print('DATA', train_x, train_y)
#print('TX', train_x, train_y)
tloss = temb.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False)
#tloss = temb.train(hp, model_design, train_x, train_y, data_dir, None, data, reg=None, emb=False)
#tloss = cv.train(hp, model_design, train_x, train_y, data_dir=data_dir, data=data, splits=splits)
#print("LOSS", tloss)
#pd.DataFrame(tloss, index=[0]).to_csv('mlp_train2.csv')


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
'''
model = models.NMLP(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
model.load_state_dict(torch.load(''.join((data_dir, 'modelev2.pth'))))
model.eval()
with torch.no_grad():
    train = model(x_train)
    test = model(x_test)
    print(mse(train, y_train))
    print(mae(test, y_test))
    print(train)
    print(test)


for i in range(splits):
    i += 1
    #import model
    model = models.NMLP(x.shape[1], y.shape[1], model_design['layersizes'])
    model.load_state_dict(torch.load(''.join((data_dir, f"mlp_model{i}.pth"))))
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



pd.DataFrame.from_dict(performance).to_csv('mlp_eval_performance2.csv')
pd.DataFrame.from_dict(preds_train).to_csv('mlp_eval_preds_train2.csv')
pd.DataFrame.from_dict(preds_test).to_csv('mlp_eval_preds_test2.csv')



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
    



    















