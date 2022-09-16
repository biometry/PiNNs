# !/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import random
import torch
import dataset

def standardize(var, scaling=None, get_p=False):
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
    if get_p:
        return out, m, std
    else:
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


def read_in(type, data_dir=None, data_use=None, exp=None):
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
        out = pd.read_csv(''.join((data_dir, 'hyytialaNAS.csv')))
        out = out[pd.DatetimeIndex(out['date']).year.isin([2004,2005])]
    elif type == 'NASp' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'hyytialaNAS.csv')))
    elif type == 'validation' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'Hyytiala.csv')))
        out = out[~pd.DatetimeIndex(out['date']).year.isin([2004,2005])]
    elif type.startswith('exp2') and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'allsites.csv')))
    elif type == 'simulations' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, f'simulations_{data_use}_{exp}.csv')))
    return out



def loaddata(data_split, history, batch_size=None, dir=None, raw=False, doy=True, sparse=False, exp = None, via=False):
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

    # Define arguments for correct simulation load. Ignored if data_split != simulations.
    if sparse:
        data_use = 'sparse'
    else:
        data_use = 'full'
    if exp != 'exp2':
        exp = ''
    data = read_in(data_split, dir, data_use, exp)
    #print(data)
    rawdata = []
    if raw:
        rawdata = data.copy()
    print(data)    
    if doy:
        data['doy_sin'], data['doy_cos'] = encode_doy(data['DOY'])
    
    if data_split != 'simulations':
        date = data['date']
        print(date)
    y = data['GPP']
    
    if ypcols:
        yp = data[ypcols]
        if via:
            data, mn, std = standardize(data.drop(['CO2', 'date', 'DOY', 'GPP', 'X', 'GPPp', 'ETp', 'SWp'], axis=1), get_p=True)
        else:
            data = standardize(data.drop(['CO2', 'date', 'DOY', 'GPP', 'X', 'GPPp', 'ETp', 'SWp'], axis=1))
    elif doy:
        if data_split != 'simulations':
            yp = None
            if via:
                data, mn, std = standardize(data.drop(['CO2', 'date', 'DOY', 'GPP'], axis=1), get_p=True)
            else:
                data = standardize(data.drop(['CO2', 'date', 'DOY', 'GPP'], axis=1))
        else:
            yp = None
            if via:
                data, mn, std = standardize(data.drop(['CO2', 'DOY', 'GPP'], axis=1), get_p=True)
            else:
                data = standardize(data.drop(['CO2', 'DOY', 'GPP'], axis=1))
    else:
        yp = None
        data = data.drop(['CO2', 'GPP', 'date'], axis=1)
        data['date'] = date

    if sparse:
        if yp is not None:
            if data_split != 'simulations':
                data, y, yp, date = make_sparse(data[xcols], y, yp, date)
                rawdata = make_sparse(rawdata)
            else:
                data, y, yp, date = make_sparse(data[xcols], y, yp, date=False)
                rawdata = make_sparse(rawdata)
        else:
            if data_split != 'simulations':
                data, y, rawdata, date = make_sparse(data[xcols], y, rawdata, date=date)
            else:
                data, y, date = make_sparse(data[xcols], y, sparse=False, date=False)
    
    if history:
        x, y = add_history(data[xcols], y, history, batch_size)
    else:
        x, y = data[xcols], y
        
    if data_split != 'simulations':
        x.index = pd.DatetimeIndex(date[history:])
        y.index = pd.DatetimeIndex(date[history:])
    
    if yp is not None:
        yp = yp[history:]
        if data_split != 'exp2':
            yp.index = pd.DatetimeIndex(yp.date[history:])
        
        if via:
            out = x, y, rawdata, yp, mn, std
        else:
            out = x, y, rawdata, yp
        
    else:
        if via:
            out = x, y, rawdata, mn, std
        else:
            out = x, y, rawdata
        
    return out


def make_sparse(x, y=False, sparse=False, date=False, it=7):
    x_small = x.iloc[::it,:]
    if y is False and sparse is False and date is False:
        return x_small
    if y is not False:
        y_small = y.iloc[::it]
        if date is not False:
            date_small = date.iloc[::it]
        else:
            date_small = False
    if sparse is not False:
        yp_small = sparse.iloc[::it, :]
        return x_small, y_small, yp_small, date_small
    else:
        return x_small, y_small, date_small
