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




def train_cv(hparams, model_design, X, Y, data_dir, splits, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None, hp=False, embtp=None, sw=None):
    print("Hyperparams", hparams)
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
    mse_t = np.empty((splits, nepoch))
    mse_v = np.empty((splits, nepoch))
    if hp:
        mse_t = np.empty(nepoch)
        mse_v = np.empty(nepoch)
    elif exp == 2 and not hp:
        mae = nn.L1Loss()
        mse_v = np.empty(splits)
        mae_v = np.empty(splits)
        
    #test_x = torch.tensor(eval_set["test_x"], dtype=torch.float32)
    #test_y = torch.tensor(eval_set["test_y"], dtype=torch.float32)
    predstest = {}
    predstrain = {}
    i = 0
    #y_tests = []
    #y_preds = []

    for t_idx, v_idx in kf.split(X):
        print("FOLD", i)
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
        elif res == 2:
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
            if embtp is None:
                model = models.EMB(X.shape[1], Y.shape[1], model_design['layersizes'], 27, 1)
            else:
                model = models.EMB(X.shape[1], Y.shape[1], model_design['layersizes'], 27, 3)
        elif res == 2:
            model = models.RES(X.shape[1], Y.shape[1], model_design['layersizes'])
        else:
            model = models.NMLP(X.shape[1], Y.shape[1], model_design['layersizes'])
        print("INIMODEL", model)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr = hparams['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=40, verbose=True)
        train_loss = []
        val_loss = []
        
        for ep in range(nepoch):
            model.train()
            batch_diff = []
            batch_loss = []
            for step, train_data in enumerate(train_loader):
                xt = train_data[0]
                yt = train_data[1]
                #print("XX", xt)
                #print("YY", yt)
                #print(xt)
                if reg is not None or res == 2:
                    yp = train_data[2]
                    if exp==2 and not emb:
                        if res != 2:
                            yp = yp.unsqueeze(-1)
                    if emb:
                        xr = train_data[3]

                optimizer.zero_grad()

                # forward
                if emb:
                    y_hat, p = model(xt, xr, embtp, sw)
                elif res == 1:
                    y_hat = model(xt)
                elif res == 2:
                    y_hat = model(xt, yp)
                else:
                    y_hat = model(xt)
                if reg is not None and not emb:
                    loss = criterion(y_hat, yt) + eta*criterion(y_hat, yp)
                elif emb:
                    if embtp is None:
                        loss = criterion(y_hat, yt) + eta*criterion(p, yp)
                    else:
                        loss = criterion(y_hat, yt) + eta*criterion(p[..., 0:1], yp)
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
            scheduler.step(train_loss)
            if exp == 2 and not hp:
                mse_t[i, ep] = train_loss
                
            if hp:
                # test model
                model.eval()
                
                e_bl = []
                # deactivate autograd
                
                for step, val_sample in enumerate(val_loader):
                    x_val = val_sample[0]
                    y_val = val_sample[1]
                    if reg is not None or res == 2:
                        yp_val = val_sample[2]
                        if exp == 2 and not emb:
                            if res != 2:
                                yp_val = yp_val.unsqueeze(-1)
                        if emb:
                            xrv = val_sample[3]
                            y_hat_val, pv = model(x_val, xrv, embtp, sw)
                        elif res == 1:
                            y_hat_val = model(x_val)
                        elif res == 2:
                            print("yval")
                            y_hat_val = model(x_val, yp_val)
                        else:
                            y_hat_val = model(x_val)
                        
                    else:
                        y_hat_val = model(x_val)

                    if reg is not None and not emb:
                        loss = criterion(y_hat_val, y_val) + eta*criterion(y_hat_val, yp_val)
                    elif reg is not None and emb:
                        if embtp is None:
                            loss = criterion(y_hat_val, y_val) + eta*criterion(pv, yp_val)
                        else:
                            loss = criterion(y_hat_val, y_val) + eta*criterion(pv[..., 0:1], yp_val)
                    else:
                        loss = criterion(y_hat_val, y_val)
                        
                    print("eval_loss", loss)
                    e_bl.append(loss.item())
                    
            val_loss = (sum(e_bl) / len(e_bl))
            if not hp:
                mse_t[i, ep] = train_loss
                mse_v[i, ep] = val_loss
            else:
                mse_t[ep] = train_loss
                mse_v[ep] = val_loss
                    
                    
            print('EPOCH', ep)
            #print('MSE V', mse_v, 'MSE T', mse_t)
                
                
        if hp:
            print("HP")
            return {'train_loss': mse_t, 'val_loss':mse_v}
        elif exp == 2 and not hp:
            model.eval()
            y_hat_val = model(x_val)
            y_hat_t = model(x_train)
            predstrain = {f"site{i}": y_hat_t.detach().flatten().numpy()}
            predstest = {f"site{i}": y_hat_val.detach().flatten().numpy()}
            pd.DataFrame.from_dict(predstrain).to_csv(f'2{data}train{i}.csv')
            pd.DataFrame.from_dict(predstest).to_csv(f'2{data}test{i}.csv')
        mse_v[i] = criterion(y_hat_val, y_val)
        mae_v[i] = mae(y_hat_val, y_val)
            
        i += 1
            
        #pd.DataFrame({'train_loss': mse_t, 'val_loss':mse_v}, index=[0]).to_csv(f"{data}_NAS_model{i}.csv")
        if exp != 2:
            torch.save(model.state_dict(), os.path.join(data_dir, f"{data}_model{i}.pth"))
        elif exp == 2:
            torch.save(model.state_dict(), os.path.join(data_dir, f"2{data}_model{i}.pth"))
                
    if exp == 2 and not hp:
        td = {}
        for i in range(splits):
            td.update({f"site{i}": mse_t[i, :]})
            
            out =  td, {"val_mse": mse_v, "val_mae": mae_v}
        else:
            out = {"train_loss": mse_t, "val_loss": mse_v}
                    
    return out

        

def finetune(hparams, model_design, train, val, data_dir, data, reg=None, emb=False, raw=None, res=None, ypreles=None, exp=None):
    nepoch = hparams['epochs']
    batchsize = hparams['batchsize']
    if reg is not None:
        eta = hparams['eta']
        
    if reg is not None:
        yp_train, yp_val = torch.tensor(reg[0].to_numpy(), dtype=torch.float32), torch.tensor(reg[1].to_numpy(), dtype=torch.float32)
        if emb:
            xr_train, xr_val = torch.tensor(raw[0].to_numpy(), dtype=torch.float32), torch.tensor(raw[1].to_numpy(), dtype=torch.float32)
    x_train, y_train = torch.tensor(train[0].to_numpy(), dtype=torch.float32), torch.tensor(train[1].to_numpy(), dtype=torch.float32)
    x_val, y_val = torch.tensor(val[0].to_numpy(), dtype=torch.float32), torch.tensor(val[1].to_numpy(), dtype=torch.float32)
    
    if reg is not None:
        if emb:
            train_set = TensorDataset(x_train, y_train, yp_train, xr_train)
            val_set = TensorDataset(x_val, y_val, yp_val, xr_val)
        else:
            train_set = TensorDataset(x_train, y_train, yp_train)
            val_set = TensorDataset(x_val, y_val, yp_val)
    elif res == 2:
        yp_train, yp_val = torch.tensor(ypreles[0].to_numpy(), dtype=torch.float32), torch.tensor(ypreles[1].to_numpy(), dtype=torch.float32)
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
        model = models.EMB(train[0].shape[1], train[1].shape[1], model_design['layersizes'], 27, 1)
    elif res == 2:
        model = models.RES(train[0].shape[1], train[1].shape[1], model_design['layersizes'])
    elif res == 1:
        model = models.NMLP(train[0].shape[1], train[1].shape[1], model_design['layersizes'])
    else:
        model = models.NMLP(train[0].shape[1], train[1].shape[1], model_design['layersizes'])
    print("INIMODEL", model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = hparams['lr'])
    
    train_loss = []
    val_loss = []
    mse_t = np.empty(nepoch)
    mse_v = np.empty(nepoch)
    
    for ep in range(nepoch):
        model.train()
        batch_diff = []
        batch_loss = []
        for step, train_data in enumerate(train_loader):
            xt = train_data[0]
            yt = train_data[1]
            #print("XX", xt)
            #print("YY", yt)
            #print(xt)
            if reg is not None or res == 2:
                yp = train_data[2]
                if exp==2 and not emb:
                    if res != 2:
                        yp = yp.unsqueeze(-1)
                if emb:
                    xr = train_data[3]
                    
            optimizer.zero_grad()
            
            # forward
            if emb:
                y_hat, p = model(xt, xr, embtp, sw)
            elif res == 1:
                y_hat = model(xt)
            elif res == 2:
                y_hat = model(xt, yp)
            else:
                y_hat = model(xt)
            if reg is not None and not emb:
                loss = criterion(y_hat, yt) + eta*criterion(y_hat, yp)
            elif emb:
                print("EMBEDDING")
                if embtp is None:
                    loss = criterion(y_hat, yt) + eta*criterion(p, yp)
                else:
                    loss = criterion(y_hat, yt) + eta*criterion(p[..., 0:1], yp)
            else:
                loss = criterion(y_hat, yt)
            print('loss', loss)
            
            loss.backward()
            optimizer.step()
            
            batch_loss.append(loss.item())
            
        model.eval()
        # results per epoch
        train_loss = sum(batch_loss)/len(batch_loss)
        e_bl = []
        # deactivate autograd
        
        for step, val_sample in enumerate(val_loader):
            x_val = val_sample[0]
            y_val = val_sample[1]
            if reg is not None or res == 2:
                yp_val = val_sample[2]
                if exp == 2 and not emb:
                    if res != 2:
                        yp_val = yp_val.unsqueeze(-1)
            if emb:
                xrv = val_sample[3]
                y_hat_val, pv = model(x_val, xrv, embtp, sw)
            elif res == 1:
                y_hat_val = model(x_val)
            elif res == 2:
                print("yval")
                y_hat_val = model(x_val, yp_val)
            else:
                y_hat_val = model(x_val)
            
                
            if reg is not None and not emb:
                loss = criterion(y_hat_val, y_val) + eta*criterion(y_hat_val, yp_val)
            elif reg is not None and emb:
                if embtp is None:
                    loss = criterion(y_hat_val, y_val) + eta*criterion(pv, yp_val)
                else:
                    loss = criterion(y_hat_val, y_val) + eta*criterion(pv[..., 0:1], yp_val)
            else:
                loss = criterion(y_hat_val, y_val)
            print("val loss", loss)
            e_bl.append(loss)
        val_loss = (sum(e_bl) / len(e_bl))
        mse_t[ep] = train_loss
        mse_v[ep] = val_loss
        
        
        print('EPOCH', ep)
        #print('MSE V', mse_v, 'MSE T', mse_t)
        
    return {'train_loss': mse_t, 'val_loss': mse_v}




