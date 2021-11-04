# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import torch
import pandas as pd
import numpy as np
import random
import training



x, y, xt = utils.loaddata('NAS', 0, dir="./data/", raw=True)
yp = xt.GPPp
swmn = np.mean(xt.SWp)
swstd = np.std(xt.SWp)
xt = xt.drop(['date', 'ET', 'Unnamed: 0', 'GPP', 'SWp', 'GPPp', 'ETp'], axis=1)
ypp = yp
ypp.index = y.index
xt.index = x.index
splits = len(x.index.year.unique())
print(x.index, y.index, "INDEX", xt.index, ypp.index)

train_x, train_y, train_yp, train_xr = x[x.index.year.isin([2004, 2006])], y[y.index.year.isin([2004, 2006])].to_frame(), ypp[ypp.index.year.isin([2004, 2006])], xt[xt.index.year.isin([2004, 2006])]
test_x, test_y, test_yp, test_xr = x[x.index.year == 2008], y[y.index.year == 2008].to_frame(),  ypp[ypp.index.year == 2008], xt[xt.index.year == 2008]


train_x.index, train_y.index, train_yp.index, train_xr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(train_yp)), np.arange(0, len(train_xr))
test_x.index, test_y.index, test_yp.index, test_xr.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(test_yp)), np.arange(0, len(test_xr))

print(train_x, train_y, train_yp, train_xr)

res_as = pd.read_csv("Nemb2AS.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:2]
b = str(a.values).split("[")[-1].split("]")[0].split(",")
c = [int(bb) for bb in b]
a = res_as.loc[res_as.val_loss.idxmin()][2:3]
b = str(a.values).split("[")[-1].split("]")[0].split(",")
d = [int(bb) for bb in b]
layersizes = [c, d]
print('layersizes', layersizes)

model_design = {'layersizes': layersizes}

res_hp = pd.read_csv("Nemb2HP_m300.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:4]
b = a.to_numpy()
lrini = b[0]
bs = b[1]
eta = b[2]

lrs = []
for i in range(100):
    l = round(random.uniform(1e-8, lrini),8)
    if l not in lrs:
        lrs.append(l)

print(lrs, len(lrs))
mse_train = []
mse_val = []

for i in range(100):

    hp = {'epochs': 500,
          'batchsize': int(bs),
          'lr': lrs[i],
          'eta': eta}
    
    data_dir = "./data/"
    data = "emb2"
    loss = training.finetune(hp, model_design, (train_x, train_y), (test_x, test_y), data_dir, data, reg=(train_yp, test_yp), raw=(train_xr, test_xr) , emb=True, sw= (swmn, swstd), embtp=2)
    mse_train.append(np.mean(loss['train_loss']))
    mse_val.append(np.mean(loss['val_loss']))

df = pd.DataFrame(lrs)
df['train_loss'] = mse_train
df['val_loss'] = mse_val
print("Random hparams search best result:")
print(df.loc[[df["val_loss"].idxmin()]])
lr = lrs[df["val_loss"].idxmin()]
print("Dataframe:", df)

df.to_csv("300emb2_lr.csv")
