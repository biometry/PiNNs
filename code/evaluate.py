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

# Load hyytiala
x, y, mn, std, xt = utils.loaddata('validation', 1, dir="./data/", raw=True)

train_x = x[x.index.year != 2008]
train_y = y[y.index.year != 2008]

test_x = x[x.index.year == 2008]
test_y = y[y.index.year == 2008]
train_x.index, train_y.index = np.arange(0, len(train_x)), np.arange(0, len(train_y)) 
test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))

eval_set = {"test_x": test_x.to_numpy(), "test_y": test_y.to_numpy()}

hparams = {'lr': 1e-3,
           'batchsize': 32,
           'epochs': 5000}
model_design = {'layersizes': [4, 16]}
state = "MLPmodel.pth"

training.cv(hparams, model_design, None, train_x, train_y, eval_set, data_dir="./data/", splits=7)














'''
# split in train and test
train_x = x[x.index.year != 2012]
train_y = y[y.index.year != 2012]

test_x = x[x.index.year == 2012]
test_y = y[y.index.year == 2012]

train_set = TensorDataset(torch.tensor(train_x.to_numpy(), dtype=torch.float32), torch.tensor(train_y.to_numpy(), dtype=torch.float32))

# Initialize model
lr = 1e-3
batchsize = 32
n_epoch = 5000
model = models.NMLP(train_x.shape[1], train_y.shape[1], layersizes=[4, 16])
model.load_state_dict(torch.load("MLPmodel.pth"))


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

train_set_size = len(train_set)
sample_id = list(range(train_set_size))
train_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[:int(train_set_size // 100 * 80)])
val_sampler = torch.utils.data.sampler.SequentialSampler(sample_id[int(train_set_size // 100 * 80):])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchsize, sampler = train_sampler, shuffle=False)


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
        
        
        # zero parameter gradients
        optimizer.zero_grad()
        
        # forward
        y_hat = model(xt)
        loss = criterion(y_hat, yt)
        print('loss', loss)
        # backward
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        
    # results per epoch
    train_loss.append(sum(batch_loss)/len(batch_loss))
    print(train_loss)
    e_bl = []
    model.eval()
    with torch.no_grad():
        for step, val_sample in enumerate(val_loader):
            x_v = val_sample[0]
            y_v = val_sample[1]

            y_hat_v = model(x_v)
            eloss = criterion(y_hat_v, y_v)
            print("eval_loss", eloss)
            e_bl.append(eloss.item())

    val_loss.append(sum(e_bl)/len(e_bl))

torch.save(model.state_dict(), "MLPft.pth")
pd.DataFrame({"train": train_loss, "valid": val_loss}).to_csv("./lossEMBbNAS.csv")

'''
