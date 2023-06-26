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


def pretraining2(data_use="full", exp="exp2", of=False, v=2):
    
    ## Define splits
    splits = 5

    # Load data for pretraining
    if data_use == 'full':
        x, y, r  = utils.loaddata('simulations', 1, dir="./data/", exp=exp)
    else:
        x, y, r = utils.loaddata('simulations', 1, dir="./data/", sparse=True, exp=exp)
    y = y.to_frame()
    
    ## Split into training and test
    train_x, test_x, train_y, test_y = train_test_split(x, y)
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)
    print(type(train_x))
    print(type(train_y))
    print('SHAPES')
    train_x.index, train_y.index, test_x.index, test_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(test_x)), np.arange(0, len(test_y)) 
    ## Pretraining in n-fold CV: choose n the same as in evalmlpDA.py
    #splits = 6

    ## OR: Evaluate on observed data from Hyytiala?
    
    #test_x = x[x.index.year == 2008]
    #test_y = y[y.index.year == 2008]
    #train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
    #test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
    #print('XY: ', x, y)
    
    # Load results from NAS
    # Architecture
    if v!=2:
        res_as = pd.read_csv(f"./results/EX2_mlpAS_{data_use}.csv")
        a = res_as.loc[res_as.ind_mini.idxmin()][1:5]
        b = a.to_numpy()
        layersizes = list(b[np.isfinite(b)].astype(int))
        print('layersizes', layersizes)
        
        model_design = {'layersizes': layersizes}
    
        # Hyperparameters
        res_hp = pd.read_csv(f"./results/EX2_mlpHP_{data_use}.csv")
        a = res_hp.loc[res_hp.ind_mini.idxmin()][1:3]
        b = a.to_numpy()
        bs = b[1]
        lr = b[0]
    else:
        d = pd.read_csv(f"/scratch/project_2000527/pgnn/results/EX2mlpHP_{data_use}_new.csv")
        a = d.loc[d.ind_mini.idxmin()]
        layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
        parms = np.array(np.matrix(a.parameters)).ravel()
        lr = parms[0]
        bs = int(parms[1])
        model_design = {'layersizes': layersizes}
        print('layersizes', layersizes)
        model_design = {'layersizes': layersizes}
    
    # Learningrate
    if of:
        res_hp = pd.read_csv(f"./results/2mlp_lr_{data_use}.csv")
        a = res_hp.loc[res_hp.ind_mini.idxmin()][1:3]
        b = a.to_numpy()
        lr = b[0]
    
    # Original: Use 5000 Epochs
    eps = 5000
    hp = {'epochs': eps,
          'batchsize': int(bs),
          'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "./data/"
    data = f"mlpDA_pretrained_{data_use}_{exp}"
    #print('DATA', train_x, train_y)
    #print('TX', train_x, train_y)
    print(train_x, train_y)
    td, se, ae = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, hp=False, exp=2)
    
    print(td, se, ae)


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
        model.load_state_dict(torch.load(''.join((data_dir, f"2mlpDA_pretrained_{data_use}_{exp}_model{i}.pth"))))
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
        
    #print(preds_train)
    
    pd.DataFrame.from_dict(performance).to_csv(f'./results/mlpDA2_pretrained_eval_performance_{data_use}.csv')
    pd.DataFrame.from_dict(preds_train).to_csv(f'./results/mlpDA2_pretrained_eval_preds_train_{data_use}.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'./results/mlpDA2_pretrained_eval_preds_test_{data_use}.csv')
    
    pd.DataFrame.from_dict(td).to_csv(f'./results/mlpDA2_eval_tloss_{data_use}_{exp}.csv')
    pd.DataFrame.from_dict(se).to_csv(f'./results/mlpDA2_eval_vseloss_{data_use}_{exp}.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'./results/mlpDA2_eval_vaeloss_{data_use}_{exp}.csv')

if __name__ == '__main__':
    pretraining2(data_use=args.d)
    
    


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
    
