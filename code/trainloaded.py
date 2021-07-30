# !/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
import utils
import random
import models
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import csv


def train(hpar, model_ld, layersizes, X, Y, Xn, mean, std, data_dir='./', pt=None):
    # initialize data
    # hyperparameters
    n_epoch = hpar['epochs']
    batchsize = hpar['batchsize']
    lr = hpar['learningrate']
    
    x_train, y_train = X, Y
    xn_train = Xn
    train_set = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(xn_train, dtype=torch.float32))
    model = models.EMB(X.shape[1], Y.shape[1], layersizes, 27, 1, None)
    model.load_state_dict(torch.load(model_ld))
    
    par = [p for nm, p in model.named_parameters() if nm.startswith('parnet')]
    base_par = [p for nm, p in model.named_parameters() if nm.startswith('resnet')]
    
    optimizer = optim.Adam([{'params': model.parnet.parameters(), 'lr': 0}, {'params': model.resnet.parameters(), 'lr': lr}], betas=(0.5,0.8)) #, weight_decay=0.03)

    #print(optimizer)
    #print(lr)
    criterion = nn.MSELoss()
    #for p in model.parnet.parameters():
    #    print(p)
    train_set_size = len(train_set)
    sample_id = list(range(train_set_size))
    train_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[:int(train_set_size // 100 * 80)])
    val_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[int(train_set_size // 100 * 80):])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler = train_sampler, shuffle=False)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.001)

    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=val_sampler, shuffle=False)
    train_loss = []
    val_loss = []
    
    for i in range(n_epoch):
        model.train()
        batch_diff = []
        batch_loss = []
        for step, train_data in enumerate(train_loader):
            xt = train_data[0]
            yt = train_data[1]
            xnt = train_data[2]

            P = torch.tensor([[413.0/400,
                               0.450, 0.118, 3., 0.748464, 12.74915/12,
                               -3.566967/10, 18.4513/10, -0.136732,
                               0.033942, 0.448975, 0.500, -0.364,
                               0.33271/10, 0.857291, 0.041781,
                               0.474173, 0.278332/5, 1.5, 0.33,
                               4.824704/4, 0., 0., 180./180,
                               0., 0., 10./10
                               ]]*len(xt))
            
            # zero parameter gradients
            optimizer.zero_grad()

            # forward
            p, y_hat = model(xt, xnt, mean, std)
            #p = model(xt, xnt)
            loss = criterion(y_hat, yt) #+ 0.01*criterion(p, P)
            #loss = criterion(p, P)
            print('loss', loss)
            # backward
            loss.backward()
            #scheduler.step()
            optimizer.step()
            #scheduler.step()
            batch_loss.append(loss.item())
        
        # results per epoch
        train_loss.append(sum(batch_loss)/len(batch_loss))
        

        # test model
        model.eval()

        e_bl = []
        # deactivate autograd
        with torch.no_grad():
            for step, val_sample in enumerate(val_loader):
                x_val = val_sample[0]
                y_val = val_sample[1]
                xn_val = val_sample[2]
                Pv = torch.tensor([[413.0/400,
                                    0.450, 0.118, 3., 0.748464, 12.74915/12,
                                    -3.566967/10, 18.4513/10, -0.136732,
                                    0.033942, 0.448975, 0.500, -0.364,
                                    0.33271/10, 0.857291, 0.041781,
                                    0.474173, 0.278332/5, 1.5, 0.33,
                                    4.824704/4, 0., 0., 180./180,
                                    0., 0., 10./10
                                    ]]*len(x_val))

                #pv = model(x_val, xn_val)
                pv, y_hat_val = model(x_val, xn_val, mean, std)
                loss = criterion(y_hat_val, y_val) #+ 0.01*criterion(pv, Pv)
                #loss = criterion(pv, Pv)
                print("eval_loss", loss)
                e_bl.append(loss.item())

        val_loss.append(sum(e_bl) / len(e_bl))

    torch.save(model.state_dict(), os.path.join(data_dir, "model.pth"))
    with open("parMod.csv", "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        writer.writerow(['train_loss', 'val_loss'])
        for tl, vl in zip(train_loss, val_loss):
            writer.writerow([tl, vl])

        
    return {'train_loss': train_loss, 'val_loss': val_loss}







