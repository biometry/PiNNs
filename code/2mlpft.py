# !/usr/bin/env python
# coding: utf-8
import random
import utils
import HP
import training
import utils
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

def mlp2ft(data_use='full'):
    if data_use=='sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)
        x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        y = y.drop(pd.DatetimeIndex(['2004-01-01']))
        

    train_x, train_y = x[x.index.year==2004], y[y.index.year==2004].to_frame()
    test_x, test_y = x[x.index.year == 2005][1:], y[y.index.year == 2005][1:].to_frame()
    print(train_x, train_y, test_x)
    train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y))
    test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
    print(train_x, train_y)

    res_as = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_mlpAS_{data_use}.csv")
    a = res_as.loc[res_as.ind_mini.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))
    print('layersizes', layersizes)

    model_design = {'layersizes': layersizes}

    res_hp = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_mlpHP_{data_use}.csv")
    a = res_hp.loc[res_hp.ind_mini.idxmin()][1:3]
    b = a.to_numpy()
    lrini = b[0]
    bs = b[1]

    lrs = []
    for i in range(300):
        l = round(random.uniform(1e-8, lrini),8)
        if l not in lrs:
            lrs.append(l)

    print(lrs, len(lrs))
    mse_train = []
    mse_val = []

    for i in range(300):

        hp = {'epochs': 2000,
              'batchsize': int(bs),
              'lr': lrs[i]}
    
        data_dir = "./data/"
        data = "mlp"
        loss = training.finetune(hp, model_design, (train_x, train_y), (test_x, test_y), data_dir, data, reg=None, emb=False)
        mse_train.append(np.mean(loss['train_loss']))
        mse_val.append(np.mean(loss['val_loss']))

    df = pd.DataFrame(lrs)
    df['train_loss'] = mse_train
    df['val_loss'] = mse_val

    df["train_loss_sd"] = mse_train_sd
    df["val_loss_sd"] = mse_val_sd
    df["ind_mini"] = ((np.array(mse_val_mean)**2 + np.array(mse_val_sd)**2)/2)

    print("Random hparams search best result:")
    print(df.loc[[df["ind_mini"].idxmin()]])
    lr = lrs[df["ind_mini"].idxmin()]
    print("Dataframe:", df)
    
    df.to_csv(f"/scratch/project_2000527/pgnn/results/2mlp_lr_{data_use}.csv")


if __name__ == '__main__':
    mlp2ft(args.d)
