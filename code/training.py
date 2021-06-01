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


def train(hpar, model_design, X, Y, data_dir='./'):
    # initialize data
    # hyperparameters
    n_epoch = hpar['epochs']
    batchsize = hpar['batchsize']
    lr = hpar['learningrate']
    layersizes = model_design['layer_sizes']
    # shuffle data
    x_train, x_test, y_train, y_test = train_test_split(X, Y)
    print(layersizes)
    print(X.shape[1], Y.shape[1])
    train_set = TensorDataset(Tensor(x_train), Tensor(y_train))
    test_set = TensorDataset(Tensor(x_test), Tensor(y_test))
    model = models.NMLP(X.shape[1], Y.shape[1], layersizes)
    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.MSELoss()
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
    train_set_size = len(train_set)
    sample_id = list(range(len(train_set)))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_id[:train_set_size//2])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_id[train_set_size // 2:])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize,
                                               sampler= train_sampler, shuffle=False)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler= val_sampler,
                                             shuffle=False)

    
    train_loss = []
    val_loss = []

    for epoch in range(n_epoch):

        model.train()

        batch_diff = []
        batch_loss = []
        for step, train_sample in enumerate(train_loader):

            x_train = train_sample[0]
            y_train = train_sample[1]
            # zero parameter gradients
            optimizer.zero_grad()

            # forward
            y_hat = model(x_train)

            loss = criterion(y_hat, y_train)

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
                
                y_hat_val = model(x_val)
                loss = criterion(y_hat_val, y_val)
                e_bl.append(loss.item())

        val_loss = sum(e_bl) / len(e_bl)

        torch.save(model.state_dict(), os.path.join(data_dir, "model.pth"))

    return {'train_loss': train_loss, 'val_loss': val_loss}







