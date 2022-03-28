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

print(args)

def evalemb(data_use="full"):

    x, y, xt = utils.loaddata('validation', 0, dir="./data/", raw=True)
    yp_tr = pd.read_csv("./data/train_hyt.csv")
    yp_te = pd.read_csv("./data/test_hyt.csv")
    yp_tr.index = pd.DatetimeIndex(yp_tr['date'])
    yp_te.index = pd.DatetimeIndex(yp_te['date'])
    xt.index = pd.DatetimeIndex(xt.date)
    SW = np.concatenate((yp_tr.SWp.values, yp_te.SWp.values))
    
    swmn = np.mean(SW)
    swstd = np.std(SW)
    rtr = xt[~xt.index.year.isin([2007, 2008])]
    rte = xt[xt.index.year == 2008]
    yptr = yp_tr.drop(yp_tr.columns.difference(['GPPp']), axis=1)
    ypte = yp_te.drop(yp_te.columns.difference(['GPPp']), axis=1)
    rtr = rtr.drop(['date', 'ET'], axis=1)
    rte = rte.drop(['date', 'ET'], axis=1)
    print(rtr, rte)
    
    #yp = yptr.merge(ypte, how="outer")

    #print(len(yptr), len(ypte))
    #print(yptr, ypte)
    #yp = pd.concat([yptr, ypte])
    #print(yp)
    
    yp_tr = yptr[~yptr.index.year.isin([2007,2008])]
    yp_te = ypte
    y = y.to_frame()
    train_x = x[~x.index.year.isin([2007,2008])]
    train_y = y[~y.index.year.isin([2007,2008])]
    splits = len(train_x.index.year.unique())
    print(train_x, train_y)
    print(yp_tr)
    
    
    test_x = x[x.index.year == 2008]
    test_y = y[y.index.year == 2008]
    print(test_x)
    
    
    #print(len(x), len(y))
    splits = len(train_x.index.year.unique())
    #print(splits)
    train_x.index, train_y.index, yp_tr.index, rtr.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(yp_tr)), np.arange(0, len(rtr)) 
    test_x.index, test_y.index, yp_te.index, rte.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(yp_te)), np.arange(0, len(rte))
    print("train_x", train_x, rtr)
    
    
    res_as = pd.read_csv(f"./results/NembAS_{data_use}.csv")
    a = res_as.loc[res_as.val_loss.idxmin()][1:2]
    b = str(a.values).split("[")[-1].split("]")[0].split(",")
    c = [int(bb) for bb in b]
    a = res_as.loc[res_as.val_loss.idxmin()][2:3]
    b = str(a.values).split("[")[-1].split("]")[0].split(",")
    d = [int(bb) for bb in b]
    layersizes = [c, d]
    print('layersizes', layersizes)
    
    model_design = {'layersizes': layersizes}
    
    # used before: Nemb2HP_m300.csv
    res_hp = pd.read_csv(f"./results/NembHP_{data_use}.csv")
    a = res_hp.loc[res_hp.val_loss.idxmin()][1:4]
    b = a.to_numpy()
    lrini = b[0]
    bs = b[1]
    eta = b[2]
    
    res_hp = pd.read_csv(f"./results/emb_lr_{data_use}.csv")
    a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
    b = a.to_numpy()
    lr = b[0]
    
    hp = {'epochs': 1000,
          'batchsize': int(bs),
          'lr': lr, # 0.002,
          'eta': eta
      }
    
    print(hp)

    data_dir = "./data/"
    data = "emb1"
    tloss = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data,raw=rtr, reg=yp_tr, emb=True, ypreles=None, exp=1, embtp=2, sw= (swmn, swstd))
    #pd.DataFrame.from_dict(tloss).to_csv('res2_test.csv')
    print(tloss)
    train_loss = tloss['train_loss']
    val_loss = tloss['val_loss']
    t1 = []
    t2 = []
    t3 = []
    t4 = []
    t5 = []
    t6 = []
    for i in range(1000):
        t1.append(train_loss[0][i])
        t2.append(train_loss[1][i])
        t3.append(train_loss[2][i])
        t4.append(train_loss[3][i])
        t5.append(train_loss[4][i])
        t6.append(train_loss[5][i])
    pd.DataFrame({"f1": t1, "f2": t2, "f3":t3, "f4": t4, "f5": t5, "f6": t6}).to_csv(f'./results/Nemb_trainloss_{data_use}.csv')
    v1 = []
    v2 = []
    v3 = []
    v4 = []
    v5 = []
    v6 = []
    for i in range(1000):
        v1.append(val_loss[0][i])
        v2.append(val_loss[1][i])
        v3.append(val_loss[2][i])
        v4.append(val_loss[3][i])
        v5.append(val_loss[4][i])
        v6.append(val_loss[5][i])
        
    pd.DataFrame({"f1": v1, "f2": v2, "f3":v3, "f4": v4, "f5": v5, "f6": v6}).to_csv(f'./results/emb_vloss_{data_use}.csv')
    
    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train, rtr = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32), torch.tensor(rtr.to_numpy(), dtype=torch.float32)
    print('TYlength', len(y_train))
    x_test, y_test, rte = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32), torch.tensor(rte.to_numpy(), dtype=torch.float32)
    
    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    
    preds_tr = {}
    preds_te = {}
    parameters_tr = {}
    parameters_te = {} 
    for i in range(splits):
        i += 1
        #import model
        model = models.EMB(x_train.shape[1], y_train.shape[1], model_design['layersizes'], 27, 3)
        model.load_state_dict(torch.load(''.join((data_dir, f"emb1_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            p_train, ptr = model(x_train, rtr, tp = 2, sw = (swmn, swstd))
            p_test, pte = model(x_test, rte, tp = 2, sw = (swmn, swstd))
            
            parameters_tr.update({f'train_p{i}': ptr.flatten().numpy()})
            parameters_te.update({f'test_p{i}': pte.flatten().numpy()})
            preds_tr.update({f'train_emb{i}':  p_train.flatten().numpy()})
            preds_te.update({f'test_emb{i}':  p_test.flatten().numpy()})
            
            train_rmse.append(mse(p_train, y_train).tolist())
            train_mae.append(mae(p_train, y_train).tolist())
            test_rmse.append(mse(p_test, y_test).tolist())
            test_mae.append(mae(p_test, y_test).tolist())
            
            
    performance = {'train_RMSE': train_rmse,
                   'train_MAE': train_mae,
                   'test_RMSE': test_rmse,
                   'test_mae': test_mae}
    
    
    pd.DataFrame.from_dict(performance).to_csv(f'./results/emb_eval_performance_{data_use}.csv')
    pd.DataFrame.from_dict(preds_tr).to_csv(f'./results/emb_eval_preds_train_{data_use}.csv')
    pd.DataFrame.from_dict(preds_te).to_csv(f'./results/emb_eval_preds_test_{data_use}.csv')
    pd.DataFrame.from_dict(parameters_tr).to_csv(f'./results/emb_parameters_train_{data_use}.csv')
    pd.DataFrame.from_dict(parameters_te).to_csv(f'./results/emb_parameters_test_{data_use}.csv')

if __name__ == '__main__':
    evalemb(args.d)
