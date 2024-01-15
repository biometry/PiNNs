# !/usr/bin/env python
# coding: utf-8
import torch
import pandas as pd
import numpy as np
import random
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc import utils
from misc import training

def NASSearchSpace(input_size, output_size, agrid_size, pgrid_size, max_layers, reg=False, emb=False):
    agrid = []
    if emb:
        n = 2
    else:
        n = 1
    for i in range(agrid_size):
        nets = []
        for nn in range(n):
            layer_size = []
            nlayers = random.randint(1, max_layers)
            for j in range(nlayers):
                size = random.choice([2, 4, 8, 16, 32, 64, 128, 256])
                layer_size.append(size)
            nets.append(layer_size)
        if emb and nets not in agrid:
            agrid.append(nets)
        elif not emb and layer_size not in agrid:
            agrid.append(layer_size)
    print("AS Grid", len(agrid))

    pgrid = []
    for i in range(pgrid_size):
        learning_rate = random.choice(np.round(np.linspace(1e-6, 0.1),4))
        batchsize = random.choice([2, 4, 8, 16, 32, 64])
        if reg is not False:
            r = random.choice(np.round(np.linspace(0.0001, 1.000, 200), 4))
            if [learning_rate, batchsize, r] not in pgrid:
                pgrid.append([learning_rate, batchsize, r])
        else:
            if [learning_rate, batchsize] not in pgrid:
                pgrid.append([learning_rate, batchsize])
    print('HP Grid', pgrid)
    
    return agrid, pgrid


def NASSearch(agrid,pgrid, X, Y, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None, hp=False, embtp=None, sw=None):

    df = pd.DataFrame()
    df['layersizes'] = []
    df['parameters'] = []
    df['train_loss'] = []
    df['val_loss'] = []
    df["train_loss_sd"] = []
    df["val_loss_sd"] = []
    df["ind_mini"] = []
    for i in range(len(agrid)):
        model_design = {"layersizes": agrid[i]}
        print(agrid)
        for p in range(len(pgrid)):
            if reg is not None:
                hparams = {"epochs": 200,
                           "batchsize": pgrid[p][1],
                           "lr": pgrid[p][0],
                           "eta": pgrid[p][2]
                       }
            else:
                # original epochs 200
                hparams = {"epochs": 200,
                           "batchsize": pgrid[p][1],
                           "lr": pgrid[p][0]
                       }
        
            running_losses = training.train_cv(hparams, model_design, X, Y, "../../data/" , splits, data, reg=reg, emb=emb, raw=raw, res=res, ypreles=ypreles, exp=exp, hp=hp, embtp=embtp, sw=sw) # train model
        

            
            df = df.append({'layersizes': agrid[i], 'parameters': pgrid[p], 'train_loss': np.mean(running_losses["train_loss"]), 
                            'val_loss': np.mean(running_losses["val_loss"]), 'train_loss_sd': np.std(running_losses["train_loss"]),
                            'val_loss_sd': np.std(running_losses["val_loss"]), 'ind_mini': ((np.array(np.mean(running_losses["val_loss"]))**2 + np.array(np.std(running_losses["val_loss"]))**2)/2)}, ignore_index=True)

    print(df)
    print("Random architecture search best result:")
    #print(df.loc[[df["val_loss"].idxmin() & df["val_loss_sd"].idxmin()]])
    print(df.loc[df["ind_mini"].idxmin()])
    # layersizes = grid[df["val_loss"].idxmin() &  df["val_loss_sd"].idxmin()]
    
    return df

