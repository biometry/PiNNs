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


def pretraining2(data_use="full", exp="exp2", of=False):
    
    ## Define splits
    splits = 8

    # Load data for pretraining
    if data_use == 'full':
        x, y, r  = utils.loaddata('simulations', 1, dir="/home/fr/fr_fr/fr_mw1205/physics_guided_nn/data/", exp=exp)
    else:
        x, y, r = utils.loaddata('simulations', 1, dir="/home/fr/fr_fr/fr_mw1205/physics_guided_nn/data/", sparse=True, exp=exp)
    y = y.to_frame()
    
    ## Split into training and test

    train_x, test_x, train_y, test_y = train_test_split(x, y)
    print(train_x.shape)
    print(test_x.shape)
    print(train_y.shape)
    print(test_y.shape)
    print(type(train_x))
    print(type(train_y))
    
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
    res_as = pd.read_csv(f"/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/EX2_mlpAS_{data_use}.csv")
    a = res_as.loc[res_as.val_loss.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))
    print('layersizes', layersizes)
    
    model_design = {'layersizes': layersizes}
    
    # Hyperparameters
    res_hp = pd.read_csv(f"/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/EX2_mlpHP_{data_use}.csv")
    a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
    b = a.to_numpy()
    bs = b[1]
    lr = b[0]
    
    # Learningrate
    if of:
        res_hp = pd.read_csv(f"/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/2mlp_lr_{data_use}.csv")
        a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
        b = a.to_numpy()
        lr = b[0]
    
    # Original: Use 5000 Epochs
    eps = 100
    hp = {'epochs': eps,
          'batchsize': int(bs),
          'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "/home/fr/fr_fr/fr_mw1205/physics_guided_nn/data/"
    data = f"mlpDA_pretrained_{data_use}_{exp}"
    #print('DATA', train_x, train_y)
    #print('TX', train_x, train_y)
    
    td, se, ae = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, reg=None, emb=False, hp=False, exp=2)

    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'2mlp_eval_tloss_{data_use}_{exp}.csv')
    pd.DataFrame.from_dict(se).to_csv(f'2mlp_eval_vseloss_{data_use}_{exp}.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'2mlp_eval_vaeloss_{data_use}_{exp}.csv')

if __name__ == '__main__':
    pretraining2(data_use="full")


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
    
