# !/usr/bin/env python
# coding: utf-8
# @author: Marieke Wesselkamp
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(sys.path)
from misc import utils
from misc import models
import pandas as pd
import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import torch
from pygam import GAM, s
import pickle
import argparse

parser = argparse.ArgumentParser(description='Define data usage and experiment')
parser.add_argument('-d', metavar='data', type=str, help='define data usage: full vs sparse')
parser.add_argument('-e', metavar='experiment', type=str, help='Experiment: exp1 vs exp2')
parser.add_argument('-n', metavar='number', type=int, help='number of samples')
args = parser.parse_args()


def climate_simulations(train_x, exp):

    year_new = random.sample(list(train_x['year'].unique()),1)
    if exp =='exp2':
        site_new = random.sample(list(train_x['site_num'].unique()), 1)
        doy_new = train_x.loc[train_x['site_num'] == site_new[0], 'DOY']
    else:
        doy_new = train_x.loc[train_x['year']==year_new[0], 'DOY']

    X_new = pd.DataFrame()
    X_new['DOY'] = doy_new
    X_new['year'] = year_new[0]
    if exp == 'exp2':
        X_new['site_num'] = site_new[0]
        print(X_new)
    
    #%% Predict to new data set
    with open('./results/gamTair', 'rb') as f:
        gamTair = pickle.load(f)
    with open('./results/gamPrecip', 'rb') as f:
        gamPrecip = pickle.load(f)
    with open('./results/gamVPD', 'rb') as f:
        gamVPD = pickle.load(f)
    with open('./results/gamPAR', 'rb') as f:
        gamPAR = pickle.load(f)    
    with open('./results/gamfapar', 'rb') as f:
        gamfapar = pickle.load(f)

    Tair_hat = gamTair.predict(X_new)
    Precip_hat = gamPrecip.predict(X_new)
    VPD_hat = gamVPD.predict(X_new)
    PAR_hat = gamPAR.predict(X_new)
    fapar_hat = gamfapar.predict(X_new)

    if exp == 'exp2':

        print(train_x['site_num'] == site_new[0])

        Tair_true = train_x.loc[train_x['site_num']==site_new[0],'Tair']

        print(len(Tair_true))

        Precip_true = train_x.loc[train_x['site_num'] == site_new[0], 'Precip']
        VPD_true = train_x.loc[train_x['site_num'] == site_new[0], 'VPD']
        PAR_true = train_x.loc[train_x['site_num'] == site_new[0], 'PAR']
        fapar_true = train_x.loc[train_x['site_num'] == site_new[0], 'fapar']
    else:
        Tair_true = train_x.loc[train_x['year']==year_new[0],'Tair']
        Precip_true = train_x.loc[train_x['year']==year_new[0],'Precip']
        VPD_true = train_x.loc[train_x['year'] == year_new[0], 'VPD']
        PAR_true = train_x.loc[train_x['year'] == year_new[0], 'PAR']
        fapar_true = train_x.loc[train_x['year'] == year_new[0], 'fapar']

    Tair_res = gamTair.deviance_residuals(X_new, Tair_true)
    Precip_res = gamPrecip.deviance_residuals(X_new, Precip_true)
    VPD_res = gamVPD.deviance_residuals(X_new, VPD_true)
    PAR_res = gamPAR.deviance_residuals(X_new, PAR_true)
    fapar_res = gamfapar.deviance_residuals(X_new, fapar_true)
    
    res_mat = np.transpose(np.column_stack([Tair_res, Precip_res, VPD_res, PAR_res, fapar_res]))
    cov_mat = np.cov(res_mat)
    noise = np.random.multivariate_normal([0,0,0,0,0], cov=cov_mat, size = res_mat.shape[1])
    
    Tair_hat = Tair_hat + noise[:,0]
    Precip_hat = Precip_hat + noise[:,1]
    VPD_hat = VPD_hat + noise[:,2]
    PAR_hat = PAR_hat + noise[:,3]
    fapar_hat = fapar_hat + noise[:,4]

    if exp == 'exp2':
        preds = np.column_stack([X_new['year'], X_new['DOY'], Tair_hat, Precip_hat, VPD_hat, PAR_hat, fapar_hat])
        # add site if required
        climsims = pd.DataFrame(preds, columns=['year', 'DOY', 'TAir', 'Precip', 'VPD', 'PAR', 'fAPAR'])
    else:
        preds = np.column_stack([X_new['year'], X_new['DOY'], Tair_hat, Precip_hat, VPD_hat, PAR_hat, fapar_hat])
        climsims = pd.DataFrame(preds, columns = ['year', 'DOY', 'TAir', 'Precip', 'VPD', 'PAR', 'fAPAR'])

    #climsims['year'] = climsims['year'].astype(int)
    climsims['CO2'] = 380
    climsims.drop(['year'], axis = 1, inplace=True)
    cols = ['Precip', 'TAir', 'PAR', 'VPD', 'fAPAR', 'DOY', 'CO2']
    climsims = climsims[cols]
    
    return(climsims)    


def parameter_samples(n_samples, fix_pars = False, parameters = ['beta', 'chi', 'X[0]', 'gamma', 'alpha'], datadir = ',,/../data/'):

    '''
    This function generates samples from the default parameter space of Preles in a LHS design.
    args:
        n_samples (int): how much samples from the parameters space
        parameters (list type): names of parameters to consider for sampling

    returns
        array (numpy): nrow is number of samples, ncol is 30 parameters (dropped the last two)
    '''

    out = pd.read_csv(''.join((datadir, 'parameterRanges.csv')))
    if fix_pars:
        xmin = list((out[out['name'].isin(parameters)]['def']-out[out['name'].isin(parameters)]['def']*0.05))
        xmax = list((out[out['name'].isin(parameters)]['def']+out[out['name'].isin(parameters)]['def']*0.05))
    else:
        xmin = list(out[out['name'].isin(parameters)]['min'])
        xmax = list(out[out['name'].isin(parameters)]['max'])
    xs = [list(x) for x in zip(xmin, xmax)]
    xlimits = np.array(xs)

    sampling = LHS(xlimits = xlimits, criterion='m')
    num = n_samples
    x = sampling(num)
    print("Parameter LHS samples.")
    print("Shape: ", x.shape)
    print(x)

    d = np.zeros((num,30)) 
    for i in range(x.shape[0]):
        print("Joint sample of: ")
        print(x[i,:])
        out.loc[out['name'].isin(parameters), 'def'] = x[i,:]
        #        print("Replace the following values: ")
        #        print(out['name'].isin(parameters))
        #        print("New parameter vector: ")
        #        print(out['def'])
        d[i,:] = out['def'].to_numpy()[:30]

    # d = np.array(d)
    print("Parameter array: ")
    print(d[:5, :])

    return(d)


def gen_simulations(n, fix_pars = True, data_use = 'full', exp = None, data_dir = '../../data/'):

    if exp == "exp2":
        if data_use == 'full':
            print("Load allsites in full version.")
            x, y, xt, yp  = utils.loaddata('exp2', None, dir="../../data/", raw=True, doy=True)
        else:
            print("Load allsites in sparse version.")
            x, y, xt, yp  = utils.loaddata('exp2', None, dir="../../data/", raw=True, doy=True, sparse=True)
        x = x.drop(['doy_sin', 'doy_cos'], axis=1)
        #doys = xt['DOY']
        #sites = xt['site']
        x['site'] = xt['site'].values
        x['site_num'] = xt['site'].values
        x['DOY'] = xt['DOY'].values
        print(x)

        x = x[x.index.year == 2008]
        y = y[y.index.year == 2008]
        train_x = x.drop(pd.DatetimeIndex(['2008-01-01']))
        train_y = y.drop(pd.DatetimeIndex(['2008-01-01']))
        
    else:
        exp = ''
        if data_use == 'full':
            print("Load hyytialaF in full version.")
            x, y, xt = utils.loaddata('validation', None, dir="../../data/", raw=True, doy=False)
        else:
            print("Load hyytialaF in sparse version.")
            x, y, xt = utils.loaddata('validation', None, dir="../../data/", raw=True, doy=False, sparse=True)

        y = y.to_frame()
        # Hold out a year as test data                                                                      
        train_x = x[~x.index.year.isin([2008])]
        train_y = y[~y.index.year.isin([2008])]

    #train_x['year'] = pd.DatetimeIndex(train_x['date']).year
    train_x['year'] = train_x.index.year
    #train_x = train_x.drop(['date'], axis=1)

    if exp == 'exp2':
        train_x['site_num'] = train_x['site_num'].astype(str)
        mapping = {'h':1, 'sr':2, 'bz':3, 'ly':4 ,'co':5}
        train_x = train_x.replace({'site_num':mapping})

    if exp == 'exp2':

        #train_x['site'] = pd.to_numeric(train_x['site'], errors='ignore').astype(int)
        print("TRAINX", train_x.site_num.unique())
        gamTair = GAM(s(0, by=2, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year', 'site_num']], train_x['Tair'])
        gamPrecip = GAM(s(0, by=2, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year', 'site_num']], train_x['Precip'])
        gamVPD = GAM(s(0, by=2, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year', 'site_num']],train_x['VPD'])
        gamPAR = GAM(s(0, by=2, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year', 'site_num']],train_x['PAR'])
        gamfapar = GAM(s(0, by=2, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year', 'site_num']],train_x['fapar'])

    else:
        gamTair = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['Tair'])
        gamPrecip = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']], train_x['Precip'])
        gamVPD = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['VPD'])
        gamPAR = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['PAR'])
        gamfapar = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['fapar'])

    with open('./results/gamTair', 'wb') as f:
        pickle.dump(gamTair, f)
    with open('./results/gamPrecip', 'wb') as f:
        pickle.dump(gamPrecip, f)
    with open('./results/gamVPD', 'wb') as f:
        pickle.dump(gamVPD, f)
    with open('./results/gamPAR', 'wb') as f:
        pickle.dump(gamPAR, f)
    with open('./results/gamfapar', 'wb') as f:
        pickle.dump(gamfapar, f)

    #if not fix_pars:
    p = parameter_samples(n_samples = n, fix_pars = fix_pars)
    pdd = pd.DataFrame(p)
    pdd.to_csv(f'./results/DA_parameter_samples_{n}.csv', index=False)
    #np.savetext('parameter_simulations.csv', p, delimiter=';')
    pt = torch.tensor(p, dtype=torch.float64)

    #else:
    #    if exp == 'exp2':
    #        print('exp2')
    #        p = pd.read_csv(f"./data/P_CVfit_{data_use}2.csv")
    #        p = np.mean(p.to_numpy()[:30,1:5], axis=1)
    #
    #    else:
    #        p = pd.read_csv(f"./data/P_CVfit_{data_use}.csv")
    #        p = np.mean(p.to_numpy()[:30, 1:5], axis=1)
    #    pt = torch.tensor(p, dtype=torch.float64)
    
    d = []
    #if fix_pars:
    #    for i in range(n):
    #        c = climate_simulations(train_x, exp)
    #        ct = torch.tensor(c.to_numpy(), dtype=torch.float64)
    #        print(pt)
    #        out = models.physical_forward(parameters=pt, input_data=ct)
    #        print(out)
    #        out = out.detach().numpy()
    #        c['GPP'] = out
    #        #print(c)
    #        d.append(c)
    #else:
    for i in range(n):
        c = climate_simulations(train_x, exp)
        #np.savetext('climate_simulations.csv', c.to_numpy(), delimiter=';')
        ct = torch.tensor(c.to_numpy(), dtype=torch.float64)
        out = models.physical_forward(parameters = pt[i,:], input_data=ct)
        out = out.detach().numpy()
        c['GPP'] = out
        #print(c)
        d.append(c)

    d = pd.concat(d)
    d.to_csv(f'./results/simulations_{data_use}_{exp}_{n}.csv', index=False)



''' 
In case you want to make plots on the cluster computer, save them directly to results.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(predsTair)
fig.savefig('results/temp.png')
''' 


if __name__ == '__main__':
    if args.e == 'exp1':
        exp=''
    else:
        exp='exp2'
    gen_simulations(n=args.n, fix_pars=True, data_use=args.d, exp=exp)
    #gen_simulations(n = 500, fix_pars=True, data_use='sparse', exp='')
    #gen_simulations(n=1000, fix_pars=True, data_use='full', exp='exp2')
    #gen_simulations(n=1000, fix_pars=True, data_use='sparse', exp='exp2')
    
    

