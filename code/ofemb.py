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

x, y, xt = utils.loaddata('OF', 0, dir="./data/", raw=True)
print('n', x,y,xt)
yp = pd.read_csv("./data/train_soro.csv")
yp.index = pd.DatetimeIndex(yp['date'])

xt.index = pd.DatetimeIndex(xt.date)

SW = np.concatenate((yp.SWp.values, yp.SWp.values))

swmn = np.mean(SW)
swstd = np.std(SW)
rtr = xt[xt.index.year == 2002]
rte = xt[xt.index.year == 2003]

ypg = yp.drop(yp.columns.difference(['GPPp']), axis=1)
rtr = rtr.drop(['date', 'ET', 'GPP'], axis=1)
rte = rte.drop(['date', 'ET', 'GPP'], axis=1)
print(rtr, rte)

#yp = yptr.merge(ypte, how="outer")

#print(len(yptr), len(ypte))
#print(yptr, ypte)
#yp = pd.concat([yptr, ypte])
#print(yp)


yp_tr = ypg[ypg.index.year==2002]
yp_te = ypg[ypg.index.year==2003]
y = y.to_frame()
train_x = x[x.index.year==2002]
train_y = y[y.index.year==2002]

print(train_x, train_y)
print(yp_tr)


test_x = x[x.index.year == 2003]
test_y = y[y.index.year == 2003]
print(test_x)


#print(len(x), len(y))

#print(splits)
train_x.index, train_y.index, yp_tr.index, rtr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr)), np.arange(0, len(rtr)) 
test_x.index, test_y.index, yp_te.index, rte.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(yp_te)), np.arange(0, len(rte))
print("train_x", train_x, rtr)


model_design = {'layersizes': [[128], [128]]}

hp = {'epochs': 1000,
      'batchsize': int(128),
      'lr': 0.01,
      'eta': 0.5
      }

print(hp)
print("TRAIN_TEST", train_x.shape,  test_x.shape, "END")

data_dir = "./data/"
data = "embof"
tloss = training.finetune(hp, model_design, (train_x, train_y), (test_x, test_y), data_dir, data, reg=(yp_tr, yp_te), raw=(rtr, rte) , emb=True, sw= (swmn, swstd), embtp=2, qn=False)
#pd.DataFrame.from_dict(tloss).to_csv('res2_test.csv')
print(tloss)
train_loss = tloss['train_loss']
val_loss = tloss['val_loss']

pd.DataFrame({"train_loss": train_loss, "val_loss": val_loss}).to_csv('OFemb_vloss.csv')
