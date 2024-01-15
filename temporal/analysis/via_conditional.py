# !/usr/bin/env python
# coding: utf-8
# @author: Marieke Wesselkamp
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from misc import utils
from misc import models
import torch
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Define data usage and splits')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
parser.add_argument('-m', metavar='model', type=str, help='define model: mlp, res, res2, reg, emb, da')
args = parser.parse_args()

def predict(test_x, test_y, m, data_use, yp, xt_test, current_dir=''):
    # Architecture
    if m == 'mlpDA':
        res_as = pd.read_csv(f"../nas/results/NmlpHP_{data_use}.csv")
        a = res_as.loc[res_as.ind_mini.idxmin()]
        layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
        model_design = {'layersizes': layersizes}
    elif m == 'emb':
        pass
    else:
        res_as = pd.read_csv(f"../nas/results/N{m}HP_{data_use}.csv")
        a = res_as.loc[res_as.ind_mini.idxmin()]
        layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
        model_design = {'layersizes': layersizes}


    data_dir = "../models/"
    print("DATA",test_x, test_y, yp)
    test_x, test_y = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)
    #yp_test = torch.tensor(yp.to_numpy()) #, dtype=torch.float32)
    xt_test = torch.tensor(xt_test.to_numpy(), dtype=torch.float32)

    preds_test = np.zeros((test_x.shape[0], 4))

    for i in range(4):
        i += 1
        #import model
        if m in ['mlp', 'res', 'reg', 'mlpDA']:
            model = models.NMLP(test_x.shape[1], 1, model_design['layersizes'])
        elif m == 'res2':
            model = models.RES(test_x.shape[1], 1, model_design['layersizes'])
        elif m == 'emb':
            model = models.EMB(test_x.shape[1], 1, [[32], [32]], 12, 1)

        if m =='mlpDA':
            model.load_state_dict(torch.load(''.join((data_dir, f"{m}_pretrained_{data_use}_1_trained_model{i}.pth"))))
        else:
            model.load_state_dict(torch.load(''.join((data_dir, f"{m}_{data_use}_model{i}.pth"))))

        model.eval()
        with torch.no_grad():
            if m == 'res2':
                p_test = model(test_x, yp_test)
            elif m == 'emb':
                pp_test, p_test = model(test_x, xt_test) #p_test = EMB output
            else:
                p_test = model(test_x)
            preds_test[:,i-1] = p_test.flatten().numpy()

    preds_test = np.mean(preds_test, axis=1)

    return preds_test


def via(data_use, model, prediction_scenario, current_dir =''):


    if data_use == 'sparse':
        make_sparse = True
    else:
        make_sparse = False

    x, y, xt, mn, std = utils.loaddata('validation', 1, dir='../../data'), raw=True, sparse=make_sparse, via=True)
    yp = pd.read_csv(f"../../data/hyytialaF_{data_use}.csv")
    yp.index = pd.DatetimeIndex(yp['date'])

    thresholds = {'PAR': [yp['PAR'].min(), yp['PAR'].max()],
                  'Tair': [yp['Tair'].min(), yp['Tair'].max()],
                  'VPD': [yp['VPD'].min(), yp['VPD'].max()],
                  'Precip': [yp['Precip'].min(), yp['Precip'].max()],
                  'fapar': [yp['fapar'].min(), yp['fapar'].max()],
                  'GPPp': [yp['GPPp'].min(), yp['GPPp'].max()],
                  'ETp': [yp['ETp'].min(), yp['ETp'].max()],
                  'SWp': [yp['SWp'].min(), yp['SWp'].max()]
                  }

    gridsize = 200
    
    if model == 'res2':
        yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
        test_x = x[x.index.year == 2008][1:]
        test_y = y[y.index.year == 2008][1:]
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']
    elif model == 'res':
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
    elif model == 'emb':
        xt.index = pd.DatetimeIndex(xt.date)
        xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET'], axis=1)
        test_xt = xt[xt.index.year == 2008]
        test_y = y[y.index.year == 2008]
        test_x = x[x.index.year == 2008]
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']
    elif model in ['mlp', 'reg', 'mlpDA']:
        test_x = x[x.index.year == 2008][1:]
        test_y = y[y.index.year == 2008][1:]
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']

    for v in variables:
        # Create variable values over complete ranges and standardize by full dataset
        if model == 'res':
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[''.join((v, '_x'))])/std[''.join((v, '_x'))]
        else:
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[v])/std[v]

        dat = test_x.copy()
        # Compute effect of variable at mean of 14 days around record dates for seasonal changes
        def via_subset(dat):
            mar = dat['2008-03-13':'2008-03-27']
            jun = dat['2008-06-14':'2008-06-28']
            sep = dat['2008-09-13':'2008-09-28']
            dec = dat['2008-12-14':'2008-12-28']
            days = {'mar':mar, 'jun':jun, 'sep':sep, 'dec':dec}
            return days

        days = via_subset(dat)
        days_yp = via_subset(yp)
        days_xt = via_subset(test_xt)

        for mon, df in days.items():
            out_i = []
            for i in var_range:
                df.loc[:,''.join((v, '_x'))] = i
                df.loc[:,''.join((v, '_y'))] = i

                ps = predict(df, test_y, model, data_use, days_yp[mon], days_xt[mon], current_dir=current_dir)

                out_i.append(ps)

            pd.DataFrame(out_i).to_csv(f'results/{model}_{data_use}_{v}_via_cond_{mon}.csv'))


if __name__ == '__main__':

    via('sparse', 'emb', prediction_scenario = 'temporal')
    via('full', 'emb', prediction_scenario = 'temporal')
    #via('sparse', 'embtest', prediction_scenario = 'temporal')
    #via('sparse', 'mlp', prediction_scenario = 'temporal')
    #via('full', 'mlp', prediction_scenario = 'temporal')
    #via('sparse', 'res2', prediction_scenario = 'temporal')
    #via('full', 'res2', prediction_scenario='temporal')
    #via('sparse', 'res', prediction_scenario='temporal')
    #via('full', 'res', prediction_scenario='temporal')
    #via('sparse', 'reg', prediction_scenario='temporal')
    #via('full', 'reg', prediction_scenario='temporal')
    #via('sparse', 'mlpDA', prediction_scenario='temporal')
    #via('full', 'mlpDA', prediction_scenario='temporal')
