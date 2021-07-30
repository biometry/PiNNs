# !/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import random
import torch
import dataset

def standardize(var, scaling=None):
    '''
    This function standardizes variables around the mean and the standard deviation
    :param var: two dimensional array of data points to normalize e.g. pd.DataFrame, torch.tensor
    :param scaling: other targets to normalize on
    :return: scaled variables in 2-D array
    '''
    
    if not scaling:
        if (isinstance(var, pd.DataFrame)):
            out = (var - np.mean(var)) / np.std(var)
            m = np.mean(var)
            std = np.std(var)
        elif (torch.is_tensor(var)):
            out = (var - torch.mean(var)) / torch.std(var)
            m = torch.mean(var)
            std = torch.std(var)
        else:
            out = (var - np.mean(var, axis=0)) / np.std(var, axis=0)
            m = np.mean(var, axis=0)
            std = np.std(var, axis=0)
    else:
        out = (var - scaling[0]) / scaling[1]
        m = scaling[0]
        std = scaling[1]

    return out, m, std


def encode_doy(doy):
    '''
    Encode cyclic feature as doy [1, 365] or [1, 366]
    :param doy: cyclic feature e.g. doy [1, 365]
    :return: encoded feature in sine and cosine
    '''
    normalized_doy = doy * (2. * np.pi / 365)
    return np.sin(normalized_doy), np.cos(normalized_doy)


def add_history(X, Y, history, batch_size=None):
    '''
    ref: Marieke Wesselkamp
    Mini-batches for training
    :param X: PRELES Inputs, standardized
    :param Y: Observed GPP and ET
    :param batch_size: batch size
    :param history: data points from which time scale before should be used
    :return: x and y
    '''
    if batch_size:
        subset = [j for j in random.sample(range(X.shape[0]), batch_size) if j > history]
        subset_h = [item for sublist in [list(range(j - history, j)) for j in subset] for item in sublist]
        x = np.concatenate((X.iloc[subset], X.iloc[subset_h]), axis=0)
        y = np.concatenate((Y.iloc[subset], Y.iloc[subset_h]), axis=0)

    else:
        x = X[history:]
        y = Y[history:]
        for i in range(1, history+1):
            outx = pd.merge(x, X.shift(periods=i)[history:], left_index=True, right_index=True)
            #outy = pd.merge(y, Y.shift(periods=i)[history:], left_index=True, right_index=True)
            x = outx
            # outy

    return x, y


def read_in(type, data_dir=None):
    if not data_dir:
        data_path = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/'
    elif data_dir == 'load':
        out = dataset.ProfoundData(type).__getitem__()
    # subset station
    if type == 'NAS' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'soro.csv')))
    elif type == 'NASp' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'soro_p.csv')))
    elif type == 'validation' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'hyytiala.csv')))
    return out



def loaddata(data_split, history, batch_size=None, dir=None, raw=False):
    if data_split.endswith('p'):
        xcols = ['GPPp', 'ETp']
        
    else:
        xcols = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'doy_sin', 'doy_cos']

    ycols = ['GPP']
    data = read_in(data_split, dir)
    rawdata = []
    if raw:
        rawdata = data.copy()

    data['doy_sin'], data['doy_cos'] = encode_doy(data['DOY'])
    date = data['date']
    data, mn, sd = standardize(data.drop(['CO2', 'date', 'DOY'], axis=1))

    if history:
        x, y = add_history(data[xcols], data[ycols], history, batch_size)
    else:
        x, y = data[xcols], data[ycols]


    
    x.index = pd.DatetimeIndex(date[history:])
    y.index = pd.DatetimeIndex(date[history:])
    return x, y, mn, sd, rawdata




