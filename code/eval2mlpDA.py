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
#parser.add_argument('-s', metavar='splits', type=int, help='number of splits for CV')
args = parser.parse_args()

print(args)

def eval2mlpDA(data_use="full", da=1, exp = "exp2", of=False, v=2):
    '''
    da specifies Domain Adaptation:                                                                                                                                        da = 1: using pretrained weight and fully retrain network                                                                                                 
        da = 2: retrain only last layer.
    '''

    if data_use == 'sparse':
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, sparse=True, eval=True)
    else:
        x, y, xt, yp = utils.loaddata('exp2', 1, dir="./data/", raw=True, eval=True)
    xt.index = pd.DatetimeIndex(xt['date'])
    xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET', 'X'], axis=1)[1:]
    # select NAS data
    train_x = x[(xt.index.year == 2008) & (xt.site != "h")]
    train_y = y[(xt.index.year == 2008) & (xt.site != "h")]
    
    if data_use == "full":
        train_x = train_x.drop(pd.DatetimeIndex(['2008-01-01']))
        train_y = train_y.drop(pd.DatetimeIndex(['2008-01-01']))
    else:
        train_x = train_x.drop(pd.DatetimeIndex(['2008-01-02']))
        train_y = train_y.drop(pd.DatetimeIndex(['2008-01-02']))
    print(x.index)
    
    test_x = x[(xt.index.year == 2008) & (xt.site == "h")]
    test_y = y[(xt.index.year == 2008) & (xt.site == "h")]

    #train_x = train_x.drop(['site_x', 'site_y'],axis=1)
    #test_x = test_x.drop(['site_x', 'site_y'],axis=1)
    

    splits = 4
    print(splits)
    print('TRAINTEST', train_x, train_y, test_x, test_y)
    train_y = train_y.to_frame()
    test_y = test_y.to_frame()
    train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y))

    #train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y))
    #test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))
    #print('XY: ', train_x, train_y)
    
    # Load results from NAS
    # Architecture
    if v!=2:
        res_as = pd.read_csv(f"./results/EX2_mlpAS_{data_use}.csv")
        a = res_as.loc[res_as.ind_mini.idxmin()][1:5]
        b = a.to_numpy()
        layersizes = list(b[np.isfinite(b)].astype(int))
        print('layersizes', layersizes)
        # Debugging: Try different model structure, smaller last layer size
        # layersizes[-1]=128
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
    
    # originally 5000 epochs
    hp = {'epochs': 5000,
          'batchsize': int(bs),
          'lr': lr}
    print('HYPERPARAMETERS', hp)
    data_dir = "./data/"
    data = f"mlpDA_pretrained_{data_use}_{exp}"
    
    td, se, ae  = training.train_cv(hp, model_design, train_x, train_y, data_dir, splits, data, domain_adaptation=da, reg=None, emb=False, hp=False, exp=2)

    print(td, se, ae)
    pd.DataFrame.from_dict(td).to_csv(f'2mlpDA_{data_use}_eval_tloss.csv')
    pd.DataFrame.from_dict(se).to_csv(f'2mlpDA_{data_use}_eval_vseloss.csv')
    pd.DataFrame.from_dict(ae).to_csv(f'2mlpDA_{data_use}_eval_vaeloss.csv')

    # Evaluation
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    x_train, y_train = torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32)
    x_test, y_test = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    train_rmse = []
    train_mae = []
    test_rmse = []
    test_mae = []
    print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    preds_train = {}
    preds_test = {}

    for i in range(splits):
        i += 1
        #import model
        model = models.NMLP(test_x.shape[1], 1, model_design['layersizes'])
        model.load_state_dict(torch.load(''.join((data_dir, f"2{data}_{da}_trained_model{i}.pth"))))
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


    print(preds_train)



    pd.DataFrame.from_dict(performance).to_csv(f'./results/2mlpDA{da}_eval_performance_{data_use}.csv')
    #pd.DataFrame.from_dict(preds_train).to_csv(f'/scratch/project_2000527/pgnn/results/2mlp_{data_use}_eval_preds_train.csv')
    pd.DataFrame.from_dict(preds_test).to_csv(f'./results/2mlpDA{da}_eval_preds_test_{data_use}.csv')





if __name__ == '__main__':
   eval2mlpDA(data_use=args.d, da=args.a)
   

   


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
    



    















