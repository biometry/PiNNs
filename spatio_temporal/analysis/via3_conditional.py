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

def predict(test_x, test_y, m, data_use, prediction_scenario, yp, xt_test, current_dir=''):

    """
    Function to compute predictions over test_x, for a given model, and a data scenario.
    Args:
        test_x: array. test input data
        test_y:
        m: char. model name. Need to be one of 'mlp', 'res', 'res2', 'reg', 'emb', 'da'
        data_use:
        prediction_scenario:
        yp:
        current_dir:

    Returns:

    """
    # Load Architecture for the current model
    if m == 'mlpDA':
        res_as = pd.read_csv(f"../nas/results/N2mlpHP_{data_use}.csv")
        a = res_as.loc[res_as.ind_mini.idxmin()]
        layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
        model_design = {'layersizes': layersizes}
    elif m == 'emb':
        pass
    else:
        res_as = pd.read_csv(f"../nas/results/N2{m}HP_{data_use}.csv")
        a = res_as.loc[res_as.ind_mini.idxmin()]
        layersizes = np.array(np.matrix(a.layersizes)).ravel().astype(int)
        model_design = {'layersizes': layersizes}

    data_dir = "../models/"

    test_x, test_y = torch.tensor(test_x.to_numpy(), dtype=torch.float32), torch.tensor(test_y.to_numpy(), dtype=torch.float32)
    #yp_test = torch.tensor(yp.to_numpy(), dtype=torch.float32)
    xt_test = torch.tensor(xt_test.to_numpy(), dtype=torch.float32)

    # Create empty array for predictions of size monthly samples x num fold in crossvalidation
    preds_test = np.zeros((test_x.shape[0], 4))

    for i in range(4):
        i += 1
        # Import the current model and load the trained weights
        if m in ['mlp', 'res', 'reg', 'mlpDA']:
            model = models.NMLP(test_x.shape[1], 1, model_design['layersizes'])
        elif m == 'res2':
            model = models.RES(test_x.shape[1], 1, model_design['layersizes'])
        elif m == 'emb':
            model = models.EMB(test_x.shape[1], 1, [[32], [32]], 12, 1)


        if prediction_scenario == 'exp2':
            if m =='mlpDA':
                model.load_state_dict(torch.load(''.join((data_dir, f"2{m}_pretrained_{data_use}_exp2_1_trained_model{i}.pth"))))
            else:
                model.load_state_dict(torch.load(''.join((data_dir, f"2{m}_{data_use}_model{i}.pth"))))
        else:
            if m =='mlpDA':
                model.load_state_dict(torch.load(''.join((data_dir, f"23{m}_pretrained_{data_use}_exp2_1_trained_model{i}.pth"))))
            elif m == 'emb':
                model.load_state_dict(torch.load(''.join((data_dir, f"3{m}_{data_use}_model{i}.pth"))))
            else:
                model.load_state_dict(torch.load(''.join((data_dir, f"23{m}_{data_use}_model{i}.pth"))))

        model.eval()

        # Compute predictions for the current model
        with torch.no_grad():
            if m == 'res2':
                p_test = model(test_x, yp_test)
            elif m == 'emb':
                pp_test, p_test = model(test_x, xt_test) #p_test = EMB prediction
            else:
                p_test = model(test_x)
            # add predictions to array
            preds_test[:,i-1] = p_test.flatten().numpy()

    # Compute mean of predictions across folds
    preds_test = np.mean(preds_test, axis=1)

    return preds_test


def via(data_use, model, prediction_scenario, current_dir = ''):

    if data_use == 'sparse':
        x, y, xt, yp, mn, std = utils.loaddata('exp2', 1, dir="../../data/", raw=True, sparse=True, via=True)

    else:
        x, y, xt, yp, mn, std = utils.loaddata('exp2', 1, dir="../../data/", raw=True, via=True)

    xt = xt.drop(0)
    xt.index = pd.DatetimeIndex(xt.date)
    xt = xt.drop(['date', 'year', 'GPPp', 'SWp', 'ETp', 'GPP', 'ET'], axis=1)

    if model in ['mlp','res', 'reg', 'mlpDA']:
        yp = pd.read_csv("../../data/allsitesF_{prediction_scenario}_{data_use}.csv")
        yp = yp.drop(0)
        yp.index = pd.to_datetime(yp['date'], format='%Y-%m-%d')
    print("XT", xt)
    thresholds = {'PAR': [xt['PAR'].min(), xt['PAR'].max()],
                  'Tair': [xt['Tair'].min(), xt['Tair'].max()],
                  'VPD': [xt['VPD'].min(), xt['VPD'].max()],
                  'Precip': [xt['Precip'].min(), xt['Precip'].max()],
                  # 'co2': [],
                  'fapar': [xt['fapar'].min(), xt['fapar'].max()] #,
                  #'GPPp': [xt['GPPp'].min(), xt['GPPp'].max()],
                  #'ETp': [xt['ETp'].min(), xt['ETp'].max()],
                  #'SWp': [xt['SWp'].min(), xt['SWp'].max()]
                  }
    print("END")
    if model == 'res2':

        yp.index = x.index
        xt.index = x.index

        if prediction_scenario == 'exp2':
            yp = yp[((yp.index.year == 2005) | (yp.index.year == 2008)) & (xt.site == "h").values]
            yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)
        elif prediction_scenario == 'exp3':
            yp = yp[(yp.index.year == 2008) & (xt.site == "h").values]
            yp = yp.drop(yp.columns.difference(['GPPp']), axis=1)

    if model == 'res':

        yptr = yp.drop(yp.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        if prediction_scenario == 'exp2':
            ypte = yp[((yp.index.year == 2005) | (yp.index.year == 2008)) & (yp.site == "h").values]
        elif prediction_scenario == 'exp3':
            ypte = yp[(yp.index.year == 2008) & (yp.site == "h").values]
        ypte = ypte.drop(ypte.columns.difference(['GPPp', 'ETp', 'SWp']), axis=1)
        y = ypte.drop(ypte.columns.difference(['GPP']), axis=1)
        n = [1,1]
        x_tr, n = utils.add_history(yptr, n, 1)
        x_te, n = utils.add_history(ypte, n, 1)
        x_tr, mn, std = utils.standardize(x_tr, get_p=True)
        x_te = utils.standardize(x_te, [mn, std])
        test_x = x_te
        test_y = y # [1:]

        variables = ['GPPp', 'ETp', 'SWp']

    elif model in ['mlp', 'res2', 'reg', 'mlpDA', 'emb']:

        if prediction_scenario == 'exp1':

            test_x = x[(x.index.year == 2008)][1:]
            test_y = y[(y.index.year == 2008)][1:]
            test_xt = xt[(xt.index.year == 2008)][1:]

        elif prediction_scenario == 'exp2':

            test_x = x[((x.index.year == 2005) | (x.index.year == 2008)) & (xt.site == "h").values][1:]
            test_y = y[((y.index.year == 2005) | (y.index.year == 2008)) & (xt.site == "h").values][1:]
            test_xt = xt[((xt.index.year == 2005) | (xt.index.year == 2008)) & (xt.site == "h").values][1:]

        elif prediction_scenario == 'exp3':

            test_x = x[(x.index.year == 2008) & (xt.site == "h").values][1:]
            test_y = y[(y.index.year == 2008) & (xt.site == "h").values][1:]
            test_xt = xt[(xt.index.year == 2008) & (xt.site == "h").values][1:]

        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']


    gridsize = 200

    for v in variables:
        # Create variable values over complete ranges and standardize by full dataset
        if model == 'res':
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[''.join((v, '_x'))])/std[''.join((v, '_x'))]
        else:
            var_range = (np.linspace(thresholds[v][0], thresholds[v][1], gridsize)-mn[v])/std[v]

        dat = test_x.copy()
        dat.sort_index(inplace=True)
        yp_dat = yp.copy()
        yp_dat.sort_index(inplace=True)
        xt_dat = test_xt.copy()
        xt_dat.sort_index(inplace=True)

        # Compute effect of variable at mean of 14 days around record dates for seasonal changes
        def via_subset(data):
            mar = dat['2008-03-13':'2008-03-27']
            jun = dat['2008-06-14':'2008-06-28']
            sep = dat['2008-09-13':'2008-09-28']
            dec = dat['2008-12-14':'2008-12-28']
            days = {'mar':mar, 'jun':jun, 'sep':sep, 'dec':dec}
            return days

        days = via_subset(dat)
        days_yp = via_subset(yp_dat)
        days_xt = via_subset(xt_dat)

        # Compute predictions for each seasonal subset of data frame
        for mon, df in days.items():
            out_i = []
            # Compute predictions for five different input variables
            for i in var_range:
                # Create data frame only from the current variable
                df.loc[:,''.join((v, '_x'))] = i
                df.loc[:,''.join((v, '_y'))] = i

                # Compute PGNN prediction for current variable in the current season with the current model and data scenario.
                ps = predict(df, test_y, model, data_use, prediction_scenario = prediction_scenario, yp = days_yp[mon],
                             xt_test=days_xt[mon], current_dir=current_dir)
                out_i.append(ps)

            pd.DataFrame(out_i).to_csv(f'results/{model}_{data_use}_{v}_via_cond_{mon}.csv')


if __name__ == '__main__':

    via('full', 'emb', prediction_scenario = 'exp2')

