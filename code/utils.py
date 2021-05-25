# !/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import random
import torch

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
        elif (torch.is_tensor(var)):
            out = (var - torch.mean(var)) / torch.std(var)
        else:
            out = (var - np.mean(var, axis=0)) / np.std(var, axis=0)
    else:
        out = (var - scaling[0]) / scaling[1]

    return out


def encode_doy(doy):
    '''
    Encode cyclic feature as doy [1, 365] or [1, 366]
    :param doy: cyclic feature e.g. doy [1, 365]
    :return: encoded feature in sine and cosine
    '''
    normalized_doy = doy * (2. * np.pi / 365)
    return np.sin(normalized_doy), np.cos(normalized_doy)


def add_history(X, Y, batch_size, history):
    '''
    ref: Marieke Wesselkamp
    Mini-batches for training
    :param X: PRELES Inputs, standardized
    :param Y: Observed GPP and ET
    :param batch_size: batch size
    :param history: data points from which time scale before should be used
    :return: x and y
    '''
    subset = [j for j in random.sample(range(X.shape[0]), batch_size) if j > history]
    subset_h = [item for sublist in [list(range(j - history, j)) for j in subset] for item in sublist]
    x = np.concatenate((X.iloc[subset], X.iloc[subset_h]), axis=0)
    y = np.concatenate((Y.iloc[subset], Y.iloc[subset_h]), axis=0)

    return x, y


def read_in(type, data_dir=None):
    if not data_dir:
        data_path = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/'
    else:
        data_path = data_dir
    # subset station
    if type == 'NAS':
        out = pd.read_csv(''.join((data_path, 'soro.csv')))
    return out



def DataLoader(data_split, batch_size, history, dir=None):
    xcols = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'doy_sin', 'doy_cos']
    ycols = ['GPP', 'ET']
    if data_split == 'NAS':
        data = read_in(data_split, dir)
        data['doy_sin'], data['doy_cos'] = encode_doy(data['DOY'])
        data = standardize(data.drop(['CO2', 'date', 'DOY'], axis=1))

    x, y = add_history(data[xcols], data[ycols], batch_size, history)


    return x, y





