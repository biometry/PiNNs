# !/usr/bin/env python
# coding: utf-8
import utils
import HP
import torch
import pandas as pd
import numpy as np
import random
import training
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
#parser.add_argument('-s', metavar='splits', type=int, help='define number of splits')
args = parser.parse_args()

print(args)

def res2ft(data_use="full"):

    x, y, xt = utils.loaddata('NAS', 1, dir="./data/", raw=True)
    
    ypreles = xt.drop(xt.columns.difference(['GPPp']), axis=1)[1:]
    ypreles.index = x.index
    
    train_x, train_y, train_yp = x[x.index.year.isin([2004, 2005])], y[y.index.year.isin([2004, 2005])].to_frame(), ypreles[ypreles.index.year.isin([2004, 2005])]
    test_x, test_y, test_yp = x[x.index.year == 2009], y[y.index.year == 2009].to_frame(),  ypreles[ypreles.index.year == 2009]

    train_x.index, train_y.index, train_yp.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)), np.arange(0, len(train_yp))
    test_x.index, test_y.index, test_yp.index = np.arange(0, len(test_x)), np.arange(0, len(test_y)), np.arange(0, len(test_yp))
    
    print(train_x, train_y, train_yp)
    
    res_as = pd.read_csv(f"./results/NresAS2_{data_use}.csv")
    a = res_as.loc[res_as.val_loss.idxmin()][1:5]
    b = a.to_numpy()
    layersizes = list(b[np.isfinite(b)].astype(int))
    print('layersizes', layersizes)
    
    model_design = {'layersizes': layersizes}
    
    res_hp = pd.read_csv(f"./results/NresHP2_{data_use}.csv")
    a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
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
        
        hp = {'epochs': 1000,
              'batchsize': int(bs),
              'lr': lrs[i]}
        
        data_dir = "./data/"
        data = "res2"
        loss = training.finetune(hp, model_design, (train_x, train_y), (test_x, test_y), data_dir, data, reg=None, emb=False, res=2, ypreles=(train_yp, test_yp))
        mse_train.append(np.mean(loss['train_loss']))
        mse_val.append(np.mean(loss['val_loss']))
        
    df = pd.DataFrame(lrs)
    df['train_loss'] = mse_train
    df['val_loss'] = mse_val
    print("Random hparams search best result:")
    print(df.loc[[df["val_loss"].idxmin()]])
    lr = lrs[df["val_loss"].idxmin()]
    print("Dataframe:", df)
    
    df.to_csv(f"./results/res2_lr_{data_use}.csv")

if __name__ == '__main__':
    res2ft(args.d)
