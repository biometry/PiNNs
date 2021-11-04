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

x, y, xt, yp = utils.loaddata('exp2', 0, dir="./data/", raw=True)
# select NAS data
print(x)
xt.index = x.index
x = x[x.index.year == 2008]
y = y[y.index.year == 2008]
yp = yp[yp.index.year == 2008]
xt = xt[xt.index.year == 2008]
x = x.drop(pd.DatetimeIndex(['2008-01-01']))
y = y.drop(pd.DatetimeIndex(['2008-01-01']))
yp = yp.drop(pd.DatetimeIndex(['2008-01-01']))
xt = xt.drop(pd.DatetimeIndex(['2008-01-01']))
swmn, swstd = np.mean(xt.SWp), np.std(xt.SWp)
print(xt.columns)
yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
xt = xt.drop(['date', 'ET', 'GPP', 'X', 'Unnamed: 0', 'GPPp', 'ETp', 'SWp'], axis=1)
x.index, y.index, xt.index, yp.index = np.arange(0, len(x)), np.arange(0, len(y)), np.arange(0, len(xt)), np.arange(0, len(yp))
#yp = yptr.merge(ypte, how="outer")
splits = 8
#print(len(yptr), len(ypte))
#print(yptr, ypte)
#yp = pd.concat([yptr, ypte])
res_as = pd.read_csv("EX2_emb2AS.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:2]
b = str(a.values).split("[")[-1].split("]")[0].split(",")
c = [int(bb) for bb in b]
a = res_as.loc[res_as.val_loss.idxmin()][2:3]
b = str(a.values).split("[")[-1].split("]")[0].split(",")
d = [int(bb) for bb in b]
layersizes = [c, d]
print('layersizes', layersizes)

model_design = {'layersizes': layersizes}

res_hp = pd.read_csv("EX2_emb2HP_mn300.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:4]
b = a.to_numpy()
lrini = b[0]
bs = b[1]
eta = b[2]

res_hp = pd.read_csv("300emb2_lr.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
lr = b[0]


hp = {'epochs': 3000,
      'batchsize': int(bs),
      'lr': lr,
      'eta': eta
      }

print(hp)

data_dir = "./data/"
data = "2emb_2"
td, se, ae = training.train_cv(hp, model_design, x, y.to_frame(), data_dir, splits, data,raw=xt, reg=yp, emb=True, ypreles=None, exp=2, embtp=2, sw= (swmn, swstd))
print(td, se, ae)
pd.DataFrame.from_dict(td).to_csv('2emb2_2eval_tloss.csv')
pd.DataFrame.from_dict(se).to_csv('2emb2_2eval_vseloss.csv')
pd.DataFrame.from_dict(ae).to_csv('2emb2_2eval_vaeloss.csv')




