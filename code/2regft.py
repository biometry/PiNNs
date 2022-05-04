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

def ft2reg(data_use='full'):
    if data_use=='sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True)
        yp.index = y.index
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)
        yp.index = y.index
        x = x.drop(pd.DatetimeIndex(['2004-01-01']))
        y = y.drop(pd.DatetimeIndex(['2004-01-01']))
        yp = yp.drop(pd.DatetimeIndex(['2004-01-01']))

    yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)

    train_yp = yp[yp.index.year == 2004]
    train_x = x[x.index.year == 2004]
    train_y = y[y.index.year == 2004]
    test_yp = yp[yp.index.year == 2005][1:]
    test_x = x[x.index.year == 2005][1:]
    test_y = y[y.index.year == 2005][1:]
    print("TRAINTEST",train_x, test_x)

    train_x.index = np.arange(0, len(train_x))
    train_y.index = np.arange(0, len(train_y))
    train_yp.index = np.arange(0, len(train_yp))
    test_x.index = np.arange(0, len(test_x))
    test_y.index = np.arange(0, len(test_y))
    test_yp.index = np.arange(0, len(test_yp))
    print("TRAINTEST",train_x, test_x)

    res_as = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_regAS_{data_use}.csv")
    a = res_as.loc[res_as.ind_mini.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))
    print('layersizes', layersizes)

    model_design = {'layersizes': layersizes}

    res_hp = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2_regHP_{data_use}.csv")
    a = res_hp.loc[res_hp.ind_mini.idxmin()][1:4]
    b = a.to_numpy()
    lrini = b[0]
    bs = b[1]
    eta = b[2]

    lrs = []
    for i in range(300):
        l = round(random.uniform(1e-8, lrini),8)
        if l not in lrs:
            lrs.append(l)

    print(lrs, len(lrs))
    mse_train_mean = []
    mse_val_mean = []
    mse_train_sd = []
    mse_val_sd = []
    print("trainshape",train_x.shape, train_y.to_frame().shape)

    for i in range(len(lrs)):

        hp = {'epochs': 1000,
              'batchsize': int(bs),
              'lr': lrs[i],
              'eta': eta}
    
        data_dir = "./data/"
        data = "2reg"
        loss = training.finetune(hp, model_design, (train_x, train_y.to_frame()), (test_x, test_y.to_frame()), data_dir, data, emb=False, reg=(train_yp, test_yp))
        mse_train_mean.append(np.mean(loss['train_loss']))
        mse_val_mean.append(np.mean(loss['val_loss']))
        mse_train_sd.append(np.std(loss['train_loss']))
        mse_val_sd.append(np.std(loss['val_loss']))
    df = pd.DataFrame(lrs)
    df['train_loss'] = mse_train_mean
    df['val_loss'] = mse_val_mean
    
    df["train_loss_sd"] = mse_train_sd
    df["val_loss_sd"] = mse_val_sd
    df["ind_mini"] = ((np.array(mse_val_mean)**2 + np.array(mse_val_sd)**2)/2)

    print("Random hparams search best result:")
    print(df.loc[[df["ind_mini"].idxmin()]])
    lr = lrs[df["ind_mini"].idxmin()]
    print("Dataframe:", df)

    df.to_csv(f"/scratch/project_2000527/pgnn/results/2reg_lr_{data_use}.csv")

if __name__ == '__main__':
    ft2reg(args.d)
