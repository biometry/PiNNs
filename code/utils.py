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

    return out


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
    '''
    Available types:
        OF: data split used for overfitting experiment (ADJUST)
        NAS: data split used for neural archtecture and hyperparameter search (ADJUST)
        exp2: contains all sites for multisite calibration
        validation: hyytiala site data (CHANGE TYPE NAME)
        simulations: simulations from Preles for domain adaptation
    '''
    if not data_dir:
        data_path = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/'
    elif data_dir == 'load':
        out = dataset.ProfoundData(type).__getitem__()
    # subset station
    if type == 'OF' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'soro.csv')))
    if type == 'NAS' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'hyytialaNAS.csv')), index_col=False)
        #out = out[pd.DatetimeIndex(out['date']).year.isin([2004,2005])]
    elif type == 'NASp' and data_dir != 'load':
        #out = pd.read_csv(''.join((data_dir, 'bilykriz.csv')))
        out = pd.read_csv(''.join((data_dir, 'hyytialaNAS.csv')), index_col=False)
        #out = out[pd.DatetimeIndex(out['date']).year.isin([2004,2005])]
    elif type == 'validation' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'hyytialaF.csv')), index_col=False)
        #out = out[pd.DatetimeIndex(out['date']).year.isin([2008, 2009, 2010, 2011, 2012])]
    elif type.startswith('exp2') and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'data_exp2.csv')))
    elif type == 'simulations' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'DA_preles_sims.csv')), index_col=False)
    return out



def loaddata(data_split, history, batch_size=None, dir=None, raw=False, doy=True):

    if data_split.endswith('p'):
        xcols = ['GPPp', 'ETp', 'SWp']
        ypcols = None
    elif data_split == "exp2":
        ypcols = ['GPPp', 'ETp', 'SWp']
        xcols = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'doy_sin', 'doy_cos']
    else:
        ypcols = None
        if doy:
            if data_split != 'simulations':
                xcols = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'doy_sin', 'doy_cos']
            else:
                xcols = ['PAR', 'TAir', 'VPD', 'Precip', 'fAPAR', 'doy_sin', 'doy_cos']
        else:
            xcols = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar', 'DOY', 'date']
            
    ycols = ['GPP']
    data = read_in(data_split, dir)
    #print(data)
    rawdata = []
    if raw:
        rawdata = data.copy()
        
    if doy:
        data['doy_sin'], data['doy_cos'] = encode_doy(data['DOY'])
    
    if data_split != 'simulations':
        date = data['date']
    y = data['GPP']
    
    if ypcols:
        yp = data[ypcols]
        data = standardize(data.drop(['CO2', 'date', 'DOY', 'GPP', 'X', 'GPPp', 'ETp', 'SWp'], axis=1))
    elif doy:
        if data_split != 'simulations':
            yp = None
            data = standardize(data.drop(['CO2', 'date', 'DOY', 'GPP'], axis=1))
        else:
            yp = None
            data = standardize(data.drop(['CO2', 'DOY', 'GPP'], axis=1))
    else:
        yp = None
        data = data.drop(['CO2', 'GPP', 'date'], axis=1)
        data['date'] = date

    if history:
        #print(data, xcols)
        x, y = add_history(data[xcols], y, history, batch_size)
    else:
        x, y = data[xcols], y
        
    if data_split != 'simulations':
        x.index = pd.DatetimeIndex(date[history:])
        y.index = pd.DatetimeIndex(date[history:])
    
    if yp is not None:
        yp = yp[history:]
        yp.index = pd.DatetimeIndex(date[history:])
        out = x, y, rawdata, yp
        
    else:
        out = x, y, rawdata
        
    return out


def sparse(x, y, reg=None, it=7):

    x_small = x.iloc[::it,:]
    y_small = y.iloc[::it]

    if reg is not None:
        reg_small = reg.iloc[::it,:]
        return x_small, y_small, reg_small
    else:
        return x_small, y_small
