import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
import models
import utils

import os.path


def train(hp, model_design, X, Y, data, data_dir, splits, reg=None):
    batchsize = hp['batchsize']
    epochs = hp['epochs']
    eta = hp['eta']
    
    kf = KFold(n_splits=splits, shuffle=False)
    print(type(splits), type(epochs))
    mse_t = np.empty((splits, epochs))
    mse_v = np.empty((splits, epochs))
    print(model_design['layersizes'])
    i = 0

    for train_idx, test_idx in kf.split(X):
        
        x_train, x_test = torch.tensor(X.to_numpy()[train_idx], dtype=torch.float32), torch.tensor(X.to_numpy()[test_idx], dtype=torch.float32)
        y_train, y_test = torch.tensor(Y.to_numpy()[train_idx], dtype=torch.float32), torch.tensor(Y.to_numpy()[test_idx], dtype=torch.float32)
        
        yp_train, yp_test = torch.tensor(reg.to_numpy()[train_idx], dtype=torch.float32), torch.tensor(reg.to_numpy()[test_idx], dtype=torch.float32)
        train_set = TensorDataset(x_train, y_train, yp_train)
        test_set = TensorDataset(x_test, y_test, yp_test)
                
        
        train_set_size = len(train_set)
        sample_id = list(range(train_set_size))
        val_set_size = len(test_set)
        vs_id = list(range(val_set_size))

        train_sampler = torch.utils.data.sampler.RandomSampler(sample_id)
        val_sampler = torch.utils.data.sampler.RandomSampler(vs_id)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler=train_sampler, shuffle=False)
        val_loader = torch.utils.data.DataLoader(test_set, batch_size=batchsize, sampler=val_sampler, shuffle=False)

        model = models.NMLP(x_train.shape[1], y_train.shape[1], model_design['layersizes'])

        optimizer = optim.Adam(model.parameters(), lr = hp['lr'])
        criterion = nn.MSELoss()

        for epoch in range(epochs):

            model.train()

            bl = []
            el = []
            for st, train_data in enumerate(train_loader):
                x = train_data[0]
                y = train_data[1]

                yhat = model(x)

                yp = train_data[2]
                loss = eta*criterion(yhat, y) + (1-eta)*criterion(yhat, yp)
                optimizer.zero_grad()
                print("loss", loss)
                loss.backward()
                optimizer.step()

                bl.append(loss.item())
            el.append(sum(bl)/len(bl))

            model.eval()
            vbl = []
            vel = []
            for step, val_data in enumerate(val_loader):
                xv = val_data[0]
                yv = val_data[1]
                yv_hat = model(xv)
                ypv = val_data[2]
                vloss = eta*criterion(yv_hat, yv) + (1-eta)*criterion(yv_hat, ypv)
                vbl.append(vloss.item())
            vel.append(sum(vbl)/len(vbl))
            #print(el)
            mse_t[i, epoch] = el[-1]
            mse_v[i, epoch] = vel[-1]

        i += 1
        torch.save(model.state_dict(), os.path.join(data_dir, f"{data}_model{i}.pth"))

    return {"train_loss": mse_t, "val_loss": mse_v}
