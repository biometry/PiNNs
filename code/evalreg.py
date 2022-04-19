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
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

def evalreg(data_use='full', of=False):
    if data_use=='sparse':
        x, y, xt = utils.loaddata('validation', 1, dir="./data/", raw=True, sparse=True)
        yp_tr = utils.make_sparse(pd.read_csv("./data/train_hyt.csv"))
        yp_te = utils.make_sparse(pd.read_csv("./data/test_hyt.csv")[6:])
    else:
        x, y, xt = utils.loaddata('validation', 1, dir="./data/", raw=True)
        yp_tr = pd.read_csv("./data/train_hyt.csv")
        yp_te = pd.read_csv("./data/test_hyt.csv")


    yp_tr.index = pd.DatetimeIndex(yp_tr['date'])
    yp_te.index = pd.DatetimeIndex(yp_te['date'])
    yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp']), axis=1)
    ypte = yp_te.drop(yp_te.columns.difference(['GPPp']), axis=1)

    #yp = yptr.merge(ypte, how="outer")
    y = y.to_frame()
    #print(len(yptr), len(ypte))
    #print(yptr, ypte)
    #yp = pd.concat([yptr, ypte])
    #print(yp)

    reg_tr = yptr[~yptr.index.year.isin([2004, 2005, 2007,2008])][1:]
    reg_te = ypte[ypte.index.year == 2008][1:]

    train_x = x[~x.index.year.isin([2004, 2005, 2007,2008])][1:]
    train_y = y[~y.index.year.isin([2004, 2005, 2007,2008])][1:]

    splits = len(train_x.index.year.unique())

    test_x = x[x.index.year == 2008][1:]
    test_y = y[y.index.year == 2008][1:]

    #print(len(x), len(y))
    splits = len(train_x.index.year.unique())
    #print(splits)
    train_x.index, train_y.index, reg_tr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(reg_tr)) 
    test_x.index, test_y.index, reg_te.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(reg_te))

    # Load results from NAS
    # Architecture
    res_as = pd.read_csv(f"results/NregAS_{data_use}.csv")
    a = res_as.loc[res_as.val_loss.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))

    model_design = {'layersizes': layersizes}

    res_hp = pd.read_csv(f"results/NregHP_{data_use}.csv")
    a = res_hp.loc[res_hp.val_loss.idxmin()][1:4]
    b = a.to_numpy()
    lr = b[0]
    bs = b[1]
    eta = b[2]
    
    if of:
        res_hp = pd.read_csv("reg_lr.csv")
        a = res_hp.loc[res_hp.val_loss.idxmin()][1:2]
        b = a.to_numpy()
        lr = b[0]


    hp = {'epochs': 5000,
          'batchsize': int(bs),
          'lr': lr,
          'eta': eta}
    print('Hyperp', hp)

    data_dir = "./data/"
    data = f"reg_{data_use}"
    #tloss = cvreg.train(hp, model_design, train_x, train_y, data_dir=data_dir, splits=splits, data=data, reg=reg_tr)
    tloss = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=reg_tr, emb=False)
    print(tloss)
    #pd.DataFrame.from_dict(tloss).to_csv('reg_train.csv')
    # Evaluation
    print(tloss)
    train_loss = tloss['train_loss']
    val_loss = tloss['val_loss']
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    #t5 = []
    #t6 = []
    for i in range(5000):
        t1.append(train_loss[0][i])
        t2.append(train_loss[1][i])
        t3.append(train_loss[2][i])
        t4.append(train_loss[3][i])
        #t5.append(train_loss[4][i])
        #t6.append(train_loss[5][i])
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3, "f4": t4}).to_csv(f'results/reg_trainloss_{data_use}.csv')
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    #v5 = []
    #v6 = []
    for i in range(5000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        v4.append(val_loss[3][i])
        #v5.append(val_loss[4][i])
        #v6.append(val_loss[5][i])

    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3, "f4": v4}).to_csv(f'results/reg_vloss_{data_use}.csv')

    mse = nn.MSELoss()
    mae = nn.L1Loss()

    x_train, y_train, tr_reg = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(reg_tr.to_numpy(), dtype=torch.float32)
    x_test, y_test, te_reg = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(reg_te.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []

    preds_tr = {}
    preds_te = {}
    for i in range(splits):
        i += 1
        #import model
        model = models.NMLP(x_train.shape[1], y_train.shape[1], model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"reg_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train = model(x_train)
            p_test = model(x_test)
            preds_tr.update({f'train_reg{i}':  p_train.flatten().numpy()})
            preds_te.update({f'test_reg{i}':  p_test.flatten().numpy()})
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())


    performance = {'train_RMSE': train_rmse,
                   'train_MAE': train_mae,
                   'test_RMSE': test_rmse,
                   'test_mae': test_mae}


    print(performance)


    pd.DataFrame.from_dict(performance).to_csv(f'results/reg_eval_{data_use}_performance.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv(f'results/reg_eval_preds_{data_use}_train.csv')
    pd.DataFrame.from_dict(preds_te).to_csv(f'results/reg_eval_preds_{data_use}_test.csv')

if __name__ == '__main__':
    evalreg(args.d)



    















