# !/usr/bin/env python
# coding: utf-8
import sys, os
import os.path
sys.path.append("/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn/code")
os.chdir("/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn/code")

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
            out = (var - var.mean()) / var.std()
            m = var.mean()
            std = var.std()
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


def read_in(type, data_dir=None, data_use=None, exp=None, sparse=False, n=None):
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
    if type == 'NAS' and data_dir != 'load' and data_use != "sparse":
        out = pd.read_csv(''.join((data_dir, 'hyytialaF_full.csv')))
        print(out)
        print(out.date)
        out = out[pd.DatetimeIndex(out['date']).year.isin([2004,2005])]
    elif type == 'NAS' and data_dir != 'load' and data_use == "sparse":
        out = pd.read_csv(''.join((data_dir, 'hyytialaF_sparse.csv')))
        out = out[pd.DatetimeIndex(out['date']).year.isin([2004,2005])]

    elif type == 'NASp' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, 'hyytialaNAS.csv')))

    elif type == 'validation' and data_dir != 'load' and data_use == 'full':
        out = pd.read_csv(''.join((data_dir, 'hyytialaF_full.csv'))) #old = Hyytiala.csv
        out = out[~pd.DatetimeIndex(out['date']).year.isin([2004,2005])]
    elif type == 'validation' and data_dir != 'load' and data_use == 'sparse':
        out = pd.read_csv(''.join((data_dir, 'hyytialaF_sparse.csv'))) #old = Hyytiala.csv
        out = out[~pd.DatetimeIndex(out['date']).year.isin([2004,2005])]

    elif type.startswith('exp2') and data_dir != 'load' and data_use != "sparse":
        out = pd.read_csv(''.join((data_dir, 'allsitesF_full.csv')))
    elif type.startswith('exp2') and data_dir != 'load' and data_use == "sparse":
        out = pd.read_csv(''.join((data_dir, 'allsitesF_sparse.csv')))

    elif type == 'simulations' and data_dir != 'load':
        out = pd.read_csv(''.join((data_dir, f'simulations_{data_use}_{exp}_{n}.csv'))) #w/o n
    return out



def loaddata(data_split, history, batch_size=None, dir=None, raw=False, doy=True, sparse=False, exp = None, via=False, n=None):
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
    data = read_in(data_split, dir, data_use, exp, n=n)
    
    rawdata = []
    if raw:
        rawdata = data.copy()
    
    if doy:
        data['doy_sin'], data['doy_cos'] = encode_doy(data['DOY'])
    
    if data_split != 'simulations':
        date = data['date']
        print(date)
    y = data['GPP']
    
    if ypcols:
        yp = data[ypcols]
        if data_split == 'exp2':
            data = data.drop(['CO2', 'date', 'DOY', 'GPP', 'X', 'GPPp', 'ETp', 'SWp', 'site'], axis=1)
        else:
            data = data.drop(['CO2', 'date', 'DOY', 'GPP', 'X', 'GPPp', 'ETp', 'SWp'], axis=1)
        if via:
            data, mn, std = standardize(data, get_p=True)
        else:
            data = standardize(data)
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

    if sparse and data_split != 'validation' and data_split != 'exp2p' and data_split != 'exp2':
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
        if data_split != 'exp2' and data_split != 'exp2p':
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


def get_seasonal_data(data_use, model, prediction_scenario,
                  current_dir='/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn'):
    if prediction_scenario == 'temporal':

        if data_use == 'sparse':
            x, y, xt, mn, std = loaddata('validation', 1, dir=os.path.join(current_dir, "data/"), raw=True,
                                               sparse=True, via=True)
            if model in ['mlp', 'res', 'res2', 'reg', 'mlpDA']:
                yp = pd.read_csv(os.path.join(current_dir, "data/hyytialaF_sparse.csv"))
                yp.index = pd.DatetimeIndex(yp['date'])

        else:
            x, y, xt, mn, std = loaddata('validation', 1, dir=os.path.join(current_dir, "data/"), raw=True,
                                               via=True)
            if model in ['mlp', 'res', 'res2', 'reg', 'mlpDA']:
                yp = pd.read_csv(os.path.join(current_dir, "data/hyytialaF_full.csv"))
                yp.index = pd.DatetimeIndex(yp['date'])

    elif prediction_scenario == 'spatial':

        if data_use == 'sparse':
            x, y, xt, yp, mn, std = loaddata('exp2', 1, dir=os.path.join(current_dir, "data/"), raw=True,
                                                   sparse=True, via=True)
            if model in ['mlp', 'res', 'res2', 'reg', 'mlpDA']:
                yp = pd.read_csv(os.path.join(current_dir, "data/allsitesF_sparse.csv"))
                yp.index = pd.DatetimeIndex(yp['date'])

        else:
            x, y, xt, yp, mn, std = loaddata('exp2', 1, dir=os.path.join(current_dir, "data/"), raw=True,
                                                   via=True)
            if model in ['mlp', 'res', 'res2', 'reg', 'mlpDA']:
                yp = pd.read_csv(os.path.join(current_dir, "data/allsitesF_full.csv"))
                yp.index = pd.DatetimeIndex(yp['date'])

    thresholds = {'PAR': [yp['PAR'].min(), yp['PAR'].max()],
                  'Tair': [yp['Tair'].min(), yp['Tair'].max()],
                  'VPD': [yp['VPD'].min(), yp['VPD'].max()],
                  'Precip': [yp['Precip'].min(), yp['Precip'].max()],
                  # 'co2': [],
                  'fapar': [yp['fapar'].min(), yp['fapar'].max()],
                  'GPPp': [yp['GPPp'].min(), yp['GPPp'].max()],
                  'ETp': [yp['ETp'].min(), yp['ETp'].max()],
                  'SWp': [yp['SWp'].min(), yp['SWp'].max()]
                  }

    if model in ['mlp', 'res2', 'reg', 'mlpDA']:
        variables = ['Tair', 'VPD', 'Precip', 'PAR', 'fapar']
    elif model == 'res':
        yptr = yp.drop(yp.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        ypte = yp.drop(yp.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        y = yp.drop(yp.columns.difference(['GPP']), axis=1)
        n = [1, 1]
        x_tr, n = add_history(yptr, n, 1)
        x_te, n = add_history(ypte, n, 1)
        x_tr, mn, std = standardize(x_tr, get_p=True)
        x_te = standardize(x_te, [mn, std])
        test_x = x_te[x_te.index.year == 2008]
        test_y = y[y.index.year == 2008][1:]
        variables = ['GPPp', 'ETp', 'SWp']

    var_ranges = {}
    gridsize=200
    for v in variables:
        # Create variable values over complete ranges and standardize by full dataset
        if model == 'res':
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[''.join((v, '_x'))])/std[''.join((v, '_x'))]
        else:
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[v])/std[v]
        var_ranges[v] = var_range

    if model == 'res2':
        yptr = yp.drop(yp.columns.difference(['GPPp']), axis=1)
        ypte = yp.drop(yp.columns.difference(['GPPp']), axis=1)
        yp_tr = yptr[~yptr.index.year.isin([2004, 2005, 2007, 2008])][1:]
        yp_te = ypte[ypte.index.year == 2008][1:]
        yp = ypte
    if model == 'res':
        yptr = yp.drop(yp.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        ypte = yp.drop(yp.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        y = yp.drop(yp.columns.difference(['GPP']), axis=1)
        n = [1, 1]
        x_tr, n = add_history(yptr, n, 1)
        x_te, n = add_history(ypte, n, 1)
        x_tr, mn, std = standardize(x_tr, get_p=True)
        x_te = standardize(x_te, [mn, std])
        test_x = x_te[x_te.index.year == 2008]
        test_y = y[y.index.year == 2008][1:]
        variables = ['GPPp', 'ETp', 'SWp']
    # if model == 'reg':
    #    yptr = yp.drop(yp.columns.difference(['GPPp']), axis=1)
    #    ypte = yp.drop(yp.columns.difference(['GPPp']), axis=1)

    elif model in ['mlp', 'res2', 'reg', 'mlpDA']:

        test_x = x[x.index.year == 2008][1:]
        test_y = y[y.index.year == 2008][1:]

    dat = test_x.copy()
    if not yp is None:
        yp_dat = yp.copy()

    # Compute effect of variable at mean of 14 days around record dates for seasonal changes
    mar = dat['2008-03-13':'2008-03-27']
    jun = dat['2008-06-14':'2008-06-28']
    sep = dat['2008-09-13':'2008-09-28']
    dec = dat['2008-12-14':'2008-12-28']
    days = {'mar': mar, 'jun': jun, 'sep': sep, 'dec': dec}

    yp_mar = yp_dat['2008-03-13':'2008-03-27']
    yp_jun = yp_dat['2008-06-14':'2008-06-28']
    yp_sep = yp_dat['2008-09-13':'2008-09-28']
    yp_dec = yp_dat['2008-12-14':'2008-12-28']
    days_yp = {'mar': yp_mar, 'jun': yp_jun, 'sep': yp_sep, 'dec': yp_dec}

    return days, days_yp, var_ranges, mn, std
