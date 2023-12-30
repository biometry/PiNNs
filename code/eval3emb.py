import torch
import pandas as pd
import numpy as np
import utils
import modelstry
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
import argparse
from sklearn.model_selection import KFold
import os
from torch.autograd import Variable
import argparse


parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
args = parser.parse_args()

def eval3emb(data_use="full"):

    if data_use == "sparse":
       x,y,xt,yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True, eval=True)
       d1 = ['2005-01-05']
    else:
        x,y,xt,yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=False, eval=True)
        d1 = ['2005-01-01']
    y=y.to_frame()
    xt.index = pd.DatetimeIndex(xt['date'])
    xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET', 'X'], axis=1)[1:]

    train_x = x[(x.index.year == 2005) & (xt.site != "h")]
    train_y = y[(x.index.year == 2005) & (xt.site != "h")]
    train_xt = xt[(xt.index.year == 2005) & (xt.site != "h")]
    train_x = train_x.drop(pd.DatetimeIndex(d1))
    train_xt = train_xt.drop(pd.DatetimeIndex(d1))
    train_y = train_y.drop(pd.DatetimeIndex(d1))

    test_x = x[(x.index.year == 2008) & (xt.site == "h")]
    test_xt = xt[(xt.index.year == 2008) & (xt.site == "h")]
    test_y = y[(y.index.year == 2008) & (xt.site == "h")]

    train_xt= train_xt.drop(['site'],axis=1)
    test_xt = test_xt.drop(['site'],axis=1)
    splits = 4
    d = pd.read_csv(f"N2embHP_{data_use}_new.csv")
    a = d.loc[d.ind_mini.idxmin()]
    c1 = [int(b) for b in np.array(np.matrix(str(a.layersizes).split(",")[0][1:-1]))[0]]
    c2 = [int(b) for b in np.array(np.matrix(str(a.layersizes).split(",")[1][1:-1]))[0]]
    layersizes = [c1, c2]
    parms = np.array(np.matrix(a.parameters)).ravel()
    lr = parms[0]
    bs = int(parms[1])
    model_design = {'layersizes': layersizes}
    print('layersizes', layersizes)


    print("TRAIN DATA", train_x, train_y, train_xt)
    print("TEST DATA", test_x, test_y, test_xt)


    test_xt.index = np.arange(0, len(test_xt))
    test_x.index = np.arange(0, len(test_x))
    test_y.index = np.arange(0, len(test_y))
    train_x.index= np.arange(0, len(train_x))
    train_y.index= np.arange(0, len(train_y))
    train_xt.index=np.arange(0, len(train_xt))

    print("TRAIN DATA", train_x, train_y, train_xt)
    batchsize = 366
    hp = {'epochs': 5000,
                          'batchsize': batchsize,
                              'lr': lr}
    data_dir = "./data/"
    data = f"3emb_{data_use}"
    tloss = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=True, hp=False, raw=train_xt, exp=2)

    train_loss = tloss['train_loss']
    val_loss = tloss['val_loss']

    t1 = []
    t2 = []
    t3 = []
    t4 = []
    for i in range(5000):
        t1.append(train_loss[0][i])
        t2.append(train_loss[1][i])
        t3.append(train_loss[2][i])
        t4.append(train_loss[3][i])
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        v4.append(val_loss[3][i])

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3, "f4":v4}).to_csv(f'./results/3emb_{data_use}_vloss.csv')
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3, "f4":t4}).to_csv(f'./results/3emb_{data_use}_trainloss.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()

    x_train, y_train, xt_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(train_xt.to_numpy(), dtype=torch.float32)
    x_test, y_test, xt_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(test_xt.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []

    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models.EMB(x.shape[1], 1, layersizes, 12, 1)
        model.load_state_dict(torch.load(''.join((data_dir, f"3emb_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            ypp, p_train = model(x_train, xt_train)
            ypt, p_test = model(x_test, xt_test)
            preds_train.update({f'train_emb{i}': p_train.flatten().numpy()})
            preds_test.update({f'test_emb{i}': p_test.flatten().numpy()})
            train_rmse.append(mse(p_train.flatten(), y_train).tolist())
            train_mae.append(mae(p_train.flatten(), y_train).tolist())
            test_rmse.append(mse(p_test.flatten(), y_test).tolist())
            test_mae.append(mae(p_test.flatten(), y_test).tolist())

    performance = {'train_RMSE': train_rmse,
               'train_MAE': train_mae,
               'test_RMSE': test_rmse,
               'test_mae': test_mae}

    print(preds_train)

    pd.DataFrame.from_dict(performance).to_csv(f'./results/3emb_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_train).to_csv(f'./results/3emb_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'./results/3emb_eval_preds_{data_use}_test.csv')


if __name__ == "__main__":
    eval3emb(args.d)
