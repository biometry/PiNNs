import pandas as pd
import numpy as np
import utils
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from smt.sampling_methods import LHS
import models
import torch
from pygam import GAM, s
import pickle

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
    
    #%% Predict to new data set
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamTair', 'rb') as f:
        gamTair = pickle.load(f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamPrecip', 'rb') as f:
        gamPrecip = pickle.load(f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamVPD', 'rb') as f:
        gamVPD = pickle.load(f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamPAR', 'rb') as f:
        gamPAR = pickle.load(f)    
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamfapar', 'rb') as f:
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


def parameter_samples(n_samples, parameters = ['beta', 'chi', 'X[0]', 'gamma', 'alpha'], datadir = '~/physics_guided_nn/data/'):

    '''
    This function generates samples from the default parameter space of Preles in a LHS design.
    args:
        n_samples (int): how much samples from the parameters space
        parameters (list type): names of parameters to consider for sampling

    returns
        array (numpy): nrow is number of samples, ncol is 30 parameters (dropped the last two)
    '''

    out = pd.read_csv(''.join((datadir, 'parameterRanges.csv')))
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


def gen_simulations(n, data_use = 'full', exp = None, data_dir = '~/physics_guided_nn/data/'):

    if exp == "exp2":
        if data_use == 'full':
            print("Load allsites in full version.")
            x, y, xt, yp  = utils.loaddata('exp2', None, dir="~/physics_guided_nn/data/", raw=True, doy=True)
        else:
            print("Load allsites in sparse version.")
            x, y, xt, yp  = utils.loaddata('exp2', None, dir="~/physics_guided_nn/data/", raw=True, doy=True, sparse=True)
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
            x, y, xt = utils.loaddata('validation', None, dir="./data/", raw=True, doy=False)
        else:
            print("Load hyytialaF in sparse version.")
            x, y, xt = utils.loaddata('validation', None, dir="./data/", raw=True, doy=False, sparse=True)

        y = y.to_frame()
        # Hold out a year as test data                                                                      
        train_x = x[~x.index.year.isin([2008])]
        train_y = y[~y.index.year.isin([2008])]

    #train_x['year'] = pd.DatetimeIndex(train_x['date']).year
    train_x['year'] = train_x.index.year
    #train_x = train_x.drop(['date'], axis=1)

    print(train_x)
    print(train_x['site'].unique())
    train_x['site_num'] = train_x['site_num'].astype(str)
    mapping = {'h':1, 's':2, 'b':3, 'l':4 ,'c':5}
    train_x = train_x.replace({'site_num':mapping})
    print(train_x['site_num'].unique())
    print(train_x.dtypes)

    if exp == 'exp2':

        #train_x['site'] = pd.to_numeric(train_x['site'], errors='ignore').astype(int)
        #print(train_x['site'])
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

    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamTair', 'wb') as f:
        pickle.dump(gamTair, f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamPrecip', 'wb') as f:
        pickle.dump(gamPrecip, f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamVPD', 'wb') as f:
        pickle.dump(gamVPD, f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamPAR', 'wb') as f:
        pickle.dump(gamPAR, f)
    with open('/home/fr/fr_fr/fr_mw1205/physics_guided_nn/results/gamfapar', 'wb') as f:
        pickle.dump(gamfapar, f)

    p = parameter_samples(n_samples = n)
    pdd = pd.DataFrame(p)
    pdd.to_csv('~/physics_guided_nn/data/DA_parameter_samples.csv', index=False)
    #np.savetext('parameter_simulations.csv', p, delimiter=';')
    pt = torch.tensor(p, dtype=torch.float64)
    
    d = []

    for i in range(n):
        c = climate_simulations(train_x, exp)
        #np.savetext('climate_simulations.csv', c.to_numpy(), delimiter=';')                                                                                    
        ct = torch.tensor(c.to_numpy(), dtype=torch.float64)
        
        out = models.physical_forward(parameters = pt[i,:], input_data=ct)
        out = out.detach().numpy()
        #np.savetext('gpp_simulations.csv')
        
        c['GPP'] = out
        print(c)
        d.append(c)

    d = pd.concat(d)
    d.to_csv(''.join((data_dir, f'simulations_{data_use}_{exp}.csv')), index=False)



''' 
In case you want to make plots on the cluster computer, save them directly to results.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(predsTair)
fig.savefig('results/temp.png')
''' 


if __name__ == '__main__':
    #gen_simulations(n = 10)
    #gen_simulations(n = 10, data_use='sparse')
    gen_simulations(n=10, exp='exp2')
    gen_simulations(n=10, data_use='sparse', exp='exp2')

