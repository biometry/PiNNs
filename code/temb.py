# !/usr/bin/env python
# coding: utf-8
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold
import utils
import random
import models
import os
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor



def train_cv(hparams, model_design, X, Y, data_dir, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None, hp=False):
    nepoch = hparams['epochs']
    batchsize = hparams['batchsize']
    if reg is not None:
        eta = hparams['eta']
    #if emb:
    #    print('EMBEDDING')
    #    nepoch = 20
    kf = KFold(n_splits=splits, shuffle = False)
    kf.get_n_splits(X)

    #rmse_t = np.zeros((splits, nepoch))
    #rmse_v = np.zeros((splits, nepoch))
    mse_t = np.zeros((splits, nepoch))
    mse_v = np.zeros((splits, nepoch))

    
    #test_x = torch.tensor(eval_set["test_x"], dtype=torch.float32)
    #test_y = torch.tensor(eval_set["test_y"], dtype=torch.float32)

    i = 0
    #y_tests = []
    #y_preds = []
    
    for t_idx, v_idx in kf.split(X):
        if emb:
            xr_train, xr_val = torch.tensor(raw.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(raw.loc[v_idx].to_numpy(), dtype=torch.float32)
        x_train, x_val = torch.tensor(X.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(X.loc[v_idx].to_numpy(), dtype=torch.float32)
        y_train, y_val = torch.tensor(Y.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(Y.loc[v_idx].to_numpy(), dtype=torch.float32)
        
        if reg is not None:
            yp_train, yp_val = torch.tensor(reg.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(reg.loc[v_idx].to_numpy(), dtype=torch.float32)
            if emb:
                train_set = TensorDataset(x_train, y_train, yp_train, xr_train)
                val_set = TensorDataset(x_val, y_val, yp_val, xr_val)
            else:
                train_set = TensorDataset(x_train, y_train, yp_train)
                val_set = TensorDataset(x_val, y_val, yp_val)
        elif res == 1 or res == 2:
            yp_train, yp_val = torch.tensor(ypreles.loc[t_idx].to_numpy(), dtype=torch.float32), torch.tensor(ypreles.loc[v_idx].to_numpy(), dtype=torch.float32)
            train_set = TensorDataset(x_train, y_train, yp_train)
            val_set = TensorDataset(x_val, y_val, yp_val)
        else:
            train_set = TensorDataset(x_train, y_train)
            val_set = TensorDataset(x_val, y_val)

        train_set_size = len(train_set)
        sample_id = list(range(train_set_size))
        val_set_size = len(val_set)
        vsample_id = list(range(val_set_size))

        if emb:
            train_sampler = torch.utils.data.sampler.SequentialSampler(sample_id)
            val_sampler = torch.utils.data.sampler.SequentialSampler(vsample_id)
        else:
            train_sampler = torch.utils.data.sampler.RandomSampler(sample_id)
            val_sampler = torch.utils.data.sampler.RandomSampler(vsample_id)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=train_sampler, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=batchsize, sampler=val_sampler, shuffle=False)
        
        if emb:
            model = models.EMB(X.shape[1], Y.shape[1], model_design['layersizes'], 27, 1)
        elif res == 2:
            model = models.RES(X.shape[1], Y.shape[1], model_design['layersizes'])
        elif res == 1:
            model = models.NMLP(yp_train.shape[1], Y.shape[1], model_design['layersizes'])
        else:
            model = models.NMLP(X.shape[1], Y.shape[1], model_design['layersizes'])
        print("INIMODEL", model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = hparams['lr'])
            
        train_loss = []
        val_loss = []
    
        for ep in range(nepoch):
            model.train()
            batch_diff = []
            batch_loss = []
            for step, train_data in enumerate(train_loader):
                xt = train_data[0]
                yt = train_data[1]
                #print(xt)
                if reg is not None or res is not None:
                    yp = train_data[2]
                if emb:
                    xr = train_data[3]
                    
                optimizer.zero_grad()
                
                # forward
                if emb:
                    y_hat, p = model(xt, xr, mn, std)
                elif res == 1:
                    y_hat = model(yp)
                elif res == 2:
                    y_hat = model(xt, yp)
                else:
                    y_hat = model(xt)
                if reg is not None and not emb:
                    loss = eta*criterion(y_hat, yt) + (1-eta)*criterion(y_hat, yp)
                elif reg is not None and emb:
                    loss = eta*criterion(y_hat, yt) + (1-eta)*criterion(p, yp)
                else:
                    loss = criterion(y_hat, yt)
                print('loss', loss)
                # backward
                loss.backward()
                optimizer.step()
                
                batch_loss.append(loss.item())

            model.eval()    
            # results per epoch
            train_loss = sum(batch_loss)/len(batch_loss)
            if exp == 2:
                mse_t[i, ep] = train_loss
                
            if exp != 2:
                # test model
                model.eval()
                
                e_bl = []
                # deactivate autograd
                with torch.no_grad():
                    for step, val_sample in enumerate(val_loader):
                        x_val = val_sample[0]
                        y_val = val_sample[1]
                        if reg is not None or res is not None:
                            yp_val = val_sample[2]
                        if emb:
                            xrv = val_sample[3]
                            y_hat_val, pv = model(x_val, xrv, mn, std)
                        elif res == 1:
                            y_hat_val = model(yp_val)
                        elif res == 2:
                            y_hat_val = model(x_val, y_val)
                        else:
                            y_hat_val = model(x_val)

                        if reg is not None and not emb:    
                            loss = eta*criterion(y_hat_val, y_val) + (1-eta)*criterion(y_hat_val, yp_val)
                        elif reg is not None and emb:
                            loss = eta*criterion(y_hat_val, y_val) + (1-eta)*criterion(pv, yp_val)
                        else:
                            loss = criterion(y_hat_val, y_val)

                        print("eval_loss", loss)
                        e_bl.append(loss.item())
                        
                    val_loss = (sum(e_bl) / len(e_bl))
                    mse_t[i, ep] = train_loss
                    mse_v[i, ep] = val_loss
        
                    
            
            if exp == 2:
                model.eval()
                with torch.no_grad():
                    if reg is not None or res is not None:
                        yp_val = val_sample[2]
                        if emb:
                            xrv = val_sample[3]
                            y_hat_val, pv = model(x_val, xrv, mn, std)
                    elif res == 1:
                        y_hat_val = model(yp_val)
                    elif res == 2:
                        y_hat_val = model(x_val, y_val)
                    else:
                        y_hat_val = model(x_val)
                
                    if reg is not None and not emb:
                        loss = eta*criterion(y_hat_val, y_val) + (1-eta)*criterion(y_hat_val, yp_val)
                    elif reg is not None and emb:
                        loss = eta*criterion(y_hat_val, y_val) + (1-eta)*criterion(pv, yp_val)
                    else:
                        loss = criterion(y_hat_val, y_val)
                
                mse_v[i, ep] = loss
        print('EPOCH', ep)
        print('MSE V', mse_v, 'MSE T', mse_t)
        

        if exp == 2 and hp:
            return {'train_loss': mse_t, 'val_loss':mse_v}
        else:
            out =  {'train_loss': mse_t, 'val_loss':mse_v}
        i += 1
        
        #pd.DataFrame({'train_loss': mse_t, 'val_loss':mse_v}, index=[0]).to_csv(f"{data}_NAS_model{i}.csv")
        torch.save(model.state_dict(), os.path.join(data_dir, f"{data}_model{i}.pth"))

    return out



def train(hpar, model_design, X, Y, data_dir, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None):
    # initialize data
    # hyperparameters
    print('----NOCV-----')
    n_epoch = hpar['epochs']
    batchsize = hpar['batchsize']
    lr = hpar['lr']
    layersizes = model_design['layersizes']
    print("reg", reg)
    print("res", res)
    if reg is not None or emb:
        eta = hpar['eta']

    if emb:
        xr_train = torch.tensor(raw.to_numpy(), dtype=torch.float32)
    x_train = torch.tensor(X.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(Y.to_numpy(), dtype=torch.float32)

    if reg is not None:
        yp_train = torch.tensor(reg.to_numpy(), dtype=torch.float32)
        if emb:
            train_set = TensorDataset(x_train, y_train, yp_train, xr_train)
            
        else:
            train_set = TensorDataset(x_train, y_train, yp_train)
            
    elif res == 1 or res == 2:
        yp_train = torch.tensor(ypreles.to_numpy(), dtype=torch.float32)
        train_set = TensorDataset(x_train, y_train, yp_train)
        
    else:
        train_set = TensorDataset(x_train, y_train)

    train_set_size = len(train_set)
    sample_id = list(range(train_set_size))

    if emb:
        train_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[:int(train_set_size// 100 * 80)])
        val_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[int(train_set_size// 100 * 80):])
    else:
        train_sampler = torch.utils.data.sampler.RandomSampler(sample_id[:int(train_set_size// 100 * 80)])
        val_sampler = torch.utils.data.sampler.RandomSampler(sample_id[int(train_set_size// 100 * 80):])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=train_sampler, shuffle=False)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=val_sampler, shuffle=False)

    if emb:
        model = models.EMB(X.shape[1], Y.shape[1], model_design['layersizes'], 27, 1)
    elif res == 2:
        model = models.RES(X.shape[1], Y.shape[1], model_design['layersizes'])
    else:
        model = models.NMLP(X.shape[1], Y.shape[1], model_design['layersizes'])
    
    print("INIMODEL", model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = hpar['lr'])

    mse_t = []
    mse_v = []

    for epoch in range(n_epoch):

        model.train()
        
        batch_diff = []
        batch_loss = []
        for step, train_sample in enumerate(train_loader):

            xt = train_sample[0]
            yt = train_sample[1]
            if emb:
                yp = train_sample[2]
                xr = train_sample[3]
            elif res is not None or reg is not None:
                yp = train_sample[2]
            
            # zero parameter gradients
            optimizer.zero_grad()
                
            if emb:
                y_hat, p = model(xt, xr)
            elif res == 2:
                y_hat = model(xt, yp)
            else:
                y_hat = model(xt)
            if reg is not None and not emb:
                loss = eta*criterion(y_hat, yt) + (1-eta)*criterion(y_hat, yp)
            elif reg is not None and emb:
                loss = eta*criterion(y_hat, yt) + (1-eta)*criterion(p, yp)
            else:
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
                if reg is not None or res is not None:
                    yp_val = val_sample[2]
                if emb:
                    xrv = val_sample[3]
                    y_hat_val, pv = model(x_val, xrv)
                elif res == 2:
                    y_hat_val = model(x_val, y_val)
                else:
                    y_hat_val = model(x_val)

                if reg is not None and not emb:
                    loss = eta*criterion(y_hat_val, y_val) + (1-eta)*criterion(y_hat_val, yp_val)
                elif reg is not None and emb:
                    loss = eta*criterion(y_hat_val, y_val) + (1-eta)*criterion(pv, yp_val)
                else:
                    loss = criterion(y_hat_val, y_val)
                print('eval loss', loss)
                e_bl.append(loss.item())
            val_loss = (sum(e_bl) / len(e_bl))

        mse_t.append(train_loss)
        mse_v.append(val_loss)
    torch.save(model.state_dict(), os.path.join(data_dir, "modelev2.pth"))

        
    return {'train_loss': mse_t, 'val_loss': mse_v}


                                                                                                                                                                                                                                                                                                                                                                                                                                                                        



