# !/usr/bin/env python
# coding: utf-8
import sys, os
import os.path
sys.path.append("/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn/code")
os.chdir("/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn/code")
import torch
import pandas as pd
import numpy as np
import utils
import models
import torch.nn as nn
import argparse

current_dir = "/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn"

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
parser.add_argument('-m', metavar='model', type=str, help='define model: mlp, res, res2, reg, emb, da')
args = parser.parse_args()

def predict(test_x, test_y, m, data_use, yp=None, current_dir=''):
    # Architecture
    if m == 'mlpDA':
        res_as = pd.read_csv(os.path.join(current_dir, f"NAS/EX2mlpHP_{data_use}_new.csv"))
    else:
        res_as = pd.read_csv(os.path.join(current_dir, f"NAS/EX2{m}HP_{data_use}_new.csv"))
    a = res_as.loc[res_as.ind_mini.idxmin()]
    layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
    parms = np.array(np.matrix(a.parameters)).ravel()

    model_design = {'layersizes': layersizes}
    data_dir = os.path.join(current_dir, "models/")
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    if m == 'res2':
        yp_test = torch.tensor(yp.to_numpy(), dtype=torch.float32)
    test_x, test_y = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)

    test_rmse = []
    test_mae = []

    preds_test = np.zeros((test_x.shape[0], 4))

    for i in range(4):
        i += 1
        #import model
        if m in ['mlp', 'res', 'reg', 'mlpDA']:
            model = models.NMLP(test_x.shape[1], 1, model_design['layersizes'])
        elif m == 'res2':
            model = models.RES(test_x.shape[1], 1, model_design['layersizes'])
        
        if m =='mlpDA':
            model.load_state_dict(torch.load(''.join((data_dir, f"2{m}_pretrained_{data_use}_exp2_1_trained_model{i}.pth"))))
        else:
            model.load_state_dict(torch.load(''.join((data_dir, f"2{m}_{data_use}_model{i}.pth"))))
        model.eval()
        with torch.no_grad():
            if m == 'res2':
                p_test = model(test_x, yp_test)
            else:
                p_test = model(test_x)
            #preds_test.update({f'test_{m}{i}': p_test.flatten().numpy()})
            preds_test[:,i-1] = p_test.flatten().numpy()

    preds_test = np.mean(preds_test, axis=1)

    return preds_test


def via(data_use, model, prediction_scenario, yp=None, current_dir = '/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn'):

    if data_use == 'sparse':
        x, y, xt, yp, mn, std = utils.loaddata('exp2', 1, dir=os.path.join(current_dir,"data/"), raw=True, sparse=True, via=True)
        if model in ['mlp', 'res', 'res2', 'reg', 'mlpDA']:
            yp = pd.read_csv(os.path.join(current_dir,"data/allsitesF_sparse.csv"))
            yp.index = pd.DatetimeIndex(yp['date'])

    else:
        x, y, xt, yp, mn, std = utils.loaddata('exp2', 1, dir=os.path.join(current_dir,"data/"), raw=True, via=True)
        if model in ['mlp','res', 'res2', 'reg', 'mlpDA']:
            yp = pd.read_csv(os.path.join(current_dir,"data/allsitesF_full.csv"))
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
        n = [1,1]
        x_tr, n = utils.add_history(yptr, n, 1)
        x_te, n = utils.add_history(ypte, n, 1)
        x_tr, mn, std = utils.standardize(x_tr, get_p=True)
        x_te = utils.standardize(x_te, [mn, std])
        test_x = x_te[x_te.index.year == 2008]
        test_y = y[y.index.year == 2008][1:]
        variables = ['GPPp', 'ETp', 'SWp']
    #if model == 'reg':
    #    yptr = yp.drop(yp.columns.difference(['GPPp']), axis=1)
    #    ypte = yp.drop(yp.columns.difference(['GPPp']), axis=1)

    elif model in ['mlp', 'res2', 'reg', 'mlpDA']:

        test_x = x[x.index.year == 2008][1:]
        test_y = y[y.index.year == 2008][1:]
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']


    gridsize = 200

    for v in variables:
        # Create variable values over complete ranges and standardize by full dataset
        if model == 'res':
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[''.join((v, '_x'))])/std[''.join((v, '_x'))]
        else:
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[v])/std[v]
        output = {'mar':None, 'jun':None, 'sep':None, 'dec':None}

        dat = test_x.copy()
        if not yp is None:
            yp_dat = yp.copy()

        # Compute effect of variable at mean of 14 days around record dates for seasonal changes
        #mar = pd.concat([dat['2008-03-13':'2008-03-27'].mean().to_frame().T] * gridsize)
        #jun = pd.concat([dat['2008-06-14':'2008-06-28'].mean().to_frame().T] * gridsize)
        #sep = pd.concat([dat['2008-09-13':'2008-09-28'].mean().to_frame().T] * gridsize)
        #dec = pd.concat([dat['2008-12-14':'2008-12-28'].mean().to_frame().T] * gridsize)
        #days = {'mar':mar, 'jun':jun, 'sep':sep, 'dec':dec}

        # Compute effect of variable at mean of 14 days around record dates for seasonal changes
        mar = dat['2008-03-13':'2008-03-27']
        jun = dat['2008-06-14':'2008-06-28']
        sep = dat['2008-09-13':'2008-09-28']
        dec = dat['2008-12-14':'2008-12-28']
        days = {'mar':mar, 'jun':jun, 'sep':sep, 'dec':dec}

        if not yp is None:
            #yp_mar = pd.concat([yp_dat['2008-03-13':'2008-03-27'].mean().to_frame().T] * gridsize)
            #yp_jun = pd.concat([yp_dat['2008-06-14':'2008-06-28'].mean().to_frame().T] * gridsize)
            #yp_sep = pd.concat([yp_dat['2008-09-13':'2008-09-28'].mean().to_frame().T] * gridsize)
            #yp_dec = pd.concat([yp_dat['2008-12-14':'2008-12-28'].mean().to_frame().T] * gridsize)
            #days_yp = {'mar':yp_mar, 'jun':yp_jun, 'sep':yp_sep, 'dec':yp_dec}

            yp_mar = yp_dat['2008-03-13':'2008-03-27']
            yp_jun = yp_dat['2008-06-14':'2008-06-28']
            yp_sep = yp_dat['2008-09-13':'2008-09-28']
            yp_dec = yp_dat['2008-12-14':'2008-12-28']
            days_yp = {'mar':yp_mar, 'jun':yp_jun, 'sep':yp_sep, 'dec':yp_dec}

        for mon, df in days.items():
            out_i = []
            for i in var_range:
                df.loc[:,''.join((v, '_x'))] = i
                df.loc[:,''.join((v, '_y'))] = i
                #test_x.index, test_y.index = np.arange(0, len(test_x)), np.arange(0, len(test_y))

                if not yp is None:
                    ps = predict(df, test_y, model, data_use, days_yp[mon], current_dir=current_dir)
                else:
                    ps = predict(df, test_y, model, data_use, yp, current_dir=current_dir)

                out_i.append(ps)

            pd.DataFrame(out_i).to_csv(os.path.join(current_dir,f'results_final/via/{prediction_scenario}/{model}_{data_use}_{v}_via_cond_{mon}.csv'))
            # pd.DataFrame.from_dict(ps).apply(lambda row: np.mean(row.to_numpy()), axis=1)


if __name__ == '__main__':
    via('sparse', 'mlp', prediction_scenario = 'spatial')
    via('full', 'mlp', prediction_scenario = 'spatial')
    via('sparse', 'res2', prediction_scenario = 'spatial')
    via('full', 'res2', prediction_scenario='spatial')
    via('sparse', 'res', prediction_scenario='spatial')
    via('full', 'res', prediction_scenario='spatial')
    via('sparse', 'reg', prediction_scenario='spatial')
    via('full', 'reg', prediction_scenario='spatial')
    via('sparse', 'mlpDA', prediction_scenario='spatial')
    via('full', 'mlpDA', prediction_scenario='spatial')
