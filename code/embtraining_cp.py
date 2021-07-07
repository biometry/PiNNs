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


def train(hpar, model_design, X, Y, Xn, data_dir='./'):
    # initialize data
    # hyperparameters
    n_epoch = hpar['epochs']
    batchsize = hpar['batchsize']
    lr = hpar['learningrate']
    layersizes = model_design['layer_sizes']
    # shuffle data
    #x_train, x_test, y_train, y_test = train_test_split(X, Y)
    print(layersizes)
    print(X.shape[1], Y.shape[1])
    x_train, y_train = X, Y
    xn_train = Xn
    print('x_train', x_train.shape, '\n y_train', y_train.shape)
    train_set = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(xn_train.to_numpy(), dtype=torch.float32))
    #test_set = TensorDataset(Tensor(x_test), Tensor(y_test))
    model = models.EMB(X.shape[1], Y.shape[1], layersizes, 30, 2)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
    train_set_size = len(train_set)
    sample_id = list(range(train_set_size))
    print("TS", train_set_size//100*80)
    print(batchsize)
    train_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[:int(train_set_size // 100 * 80)])
    val_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[int(train_set_size // 100 * 80):])
    #train_data = train_set[:int(train_set_size // 100 * 80),...]
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler = train_sampler, shuffle=False)
    
    #val_data = train_set[int(train_set_size // 100 * 80):,...]
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=val_sampler, shuffle=False)
    train_loss = []
    val_loss = []
    #nbatches = (train_set_size//100*80)/batchsize
    #print(val_loader[0:2])
    for i in range(n_epoch):
        print("model")
        model.train()
        batch_diff = []
        batch_loss = []
        for step, train_data in enumerate(train_loader):
            print(train_data)
            print(train_data[0])
            xt = train_data[0]
            yt = train_data[1]
            xnt = train_data[2]
            print(xt.shape)
            print(xt.float())

            #parameter gradients
            optimizer.zero_grad()

            # forward
            y_hat = model(xt, xnt)
            print('y_hat', y_hat)
            loss = criterion(y_hat, yt)
            print('loss', loss)
            # backward
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
        
        # results per epoch
        train_loss = sum(batch_loss)/len(batch_loss)
        

        # test model
        model.eval()

        e_bl = []
        # deactivate autograd
        with torch.no_grad():
            for step, val_sample in enumerate(val_loader):
                x_val = val_sample[0]
                y_val = val_sample[1]
                xn_val = val_sample[2]
                print(val_sample[0])
                                
                y_hat_val = model(x_val, xn_val)
                print("y_hat_val", y_hat_val)
                loss = criterion(y_hat_val, y_val)
                e_bl.append(loss.item())

        val_loss = sum(e_bl) / len(e_bl)

        torch.save(model.state_dict(), os.path.join(data_dir, "model.pth"))

    return {'train_loss': train_loss, 'val_loss': val_loss}







