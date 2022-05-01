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
parser.add_argument('-a', metavar='da', type=int, help='define type of domain adaptation')
parser.add_argument('-s', metavar='splits', type=int, help='number of splits for CV')
args = parser.parse_args()

print(args)

def evalmlpDA(data_use="full", da=3, exp = "exp2", of=False):
    '''
    da specifies Domain Adaptation:                                                                                                                                       da = 1: using pretrained weight and fully retrain network                                                                                                 
        da = 2: retrain only last layer.
    '''

    # Load hyytiala
    x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True)

    # select NAS data
    print(x.index)
    x = x[x.index.year == 2008]
    y = y[y.index.year == 2008]
    x = x.drop(pd.DatetimeIndex(['2008-01-01']))
    y = y.drop(pd.DatetimeIndex(['2008-01-01']))

    y = y.to_frame()
    splits = 8

    print(x, y)

    #train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y))
    #test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
    #print('XY: ', train_x, train_y)
    
    # Load results from NAS
    # Architecture
    res_as = pd.read_csv(f"./results/EX2_mlpAS_{data_use}.csv")
    a = res_as.loc[res_as.val_loss.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))
    print('layersizes', layersizes)
    # Debugging: Try different model structure, smaller last layer size
    # layersizes[-1]=128
    model_design = {'layersizes': layersizes}
    
    # Hyperparameters
    res_hp = pd.read_csv(f"./results/EX2_mlpHP_{data_use}.csv")
    a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
    b = a.to_numpy()
    bs = b[1]
    lr = b[0]

    # Learningrate
    if of:
        res_hp = pd.read_csv(f"./results/2mlp_lr_{data_use}.csv")
        a = res_hp.loc[res_hp.ind_mini.idxmin()][1:3]
        b = a.to_numpy()
        lr = b[0]
    
    # originally 5000 epochs
    hp = {'epochs': 100,
          'batchsize': int(bs),
          'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "./data/"
    data = f"mlpDA_pretrained_{data_use}_{exp}"
    
    td, se, ae  = training.train_cv(hp, model_design, x, y, data_dir, splits, data, domain_adaptation=da, reg=None, emb=False, hp=False, exp=2)

    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'2mlpDA_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'2mlpDA_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'2mlpDA_{data_use}_eval_vaeloss.csv')


if __name__ == '__main__':
   eval2mlpDA(args.d)


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
    



    















