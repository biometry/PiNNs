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


def train(hpar, model_design, X, Y, data_split):
    # initialize data

    n_epoch = hpar['epochs']
    n_layers = hpar['layers']

    model = torch.


    x, y = utils.DataLoader(data_split, hpar['batch_size'])
    # split data in train, test, val