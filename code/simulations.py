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


#%% load and prepare data for climate simulation
def climate_simulations(n_simulations):

    x, y, xt = utils.loaddata('validation', None, dir="~/physics_guided_nn/data/", raw=True, doy=False)
    y = y.to_frame()
    
    # Hold out a year as test data
    train_x = x[~x.index.year.isin([2012])]
    train_y = y[~y.index.year.isin([2012])]
    
    print(train_x)
    
    train_x['year'] = pd.DatetimeIndex(train_x['date']).year
    train_x = train_x.drop(['date'], axis=1)
    
    #%% Fit GAMs with cubic splines to clim ~ DOY  by year
    from pygam import GAM, s
    
    gamTair = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['Tair'])
    gamPrecip = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['Precip'])
    gamVPD = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['VPD'])
    gamPAR = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['PAR'])
    gamfapar = GAM(s(0, by=1, n_splines=200, basis='cp')).fit(train_x[['DOY', 'year']],train_x['fapar'])
    
    #print(gamTair.summary())
    #print(gamPrecip.summary())
    #print(gamVPD.summary())
    #print(gamPAR.summary())
    #print(gamfapar.summary())
    
    year_new = random.sample(list(train_x['year'].unique()),1)
    doy_new = train_x.loc[train_x['year']==year_new[0], 'DOY']
    
    X_new = pd.DataFrame()
    X_new['DOY'] = doy_new
    X_new['year'] = year_new[0]
    
    d = []
        
    for i in range(n_simulations):
        
        year_new = random.sample(list(train_x['year'].unique()),1)
        doy_new = train_x.loc[train_x['year']==year_new[0], 'DOY']
        
        X_new = pd.DataFrame()
        X_new['DOY'] = doy_new
        X_new['year'] = year_new[0]
        
        #%% Predict to new data set
        
        Tair_hat = gamTair.predict(X_new)
        Tair_true = train_x.loc[train_x['year']==year_new[0],'Tair']
        Tair_res = gamTair.deviance_residuals(X_new, Tair_true)
        
        Precip_hat = gamPrecip.predict(X_new)
        Precip_true = train_x.loc[train_x['year']==year_new[0],'Precip']
        Precip_res = gamPrecip.deviance_residuals(X_new, Precip_true)
        
        VPD_hat = gamVPD.predict(X_new)
        VPD_true = train_x.loc[train_x['year']==year_new[0],'VPD']
        VPD_res = gamVPD.deviance_residuals(X_new, VPD_true)
        
        PAR_hat = gamPAR.predict(X_new)
        PAR_true = train_x.loc[train_x['year']==year_new[0],'PAR']
        PAR_res = gamPAR.deviance_residuals(X_new, PAR_true)
        
        fapar_hat = gamfapar.predict(X_new)
        fapar_true = train_x.loc[train_x['year']==year_new[0],'fapar']
        fapar_res = gamfapar.deviance_residuals(X_new, fapar_true)
        
        res_mat = np.transpose(np.column_stack([Tair_res, Precip_res, VPD_res, PAR_res, fapar_res]))
        cov_mat = np.cov(res_mat)
        noise = np.random.multivariate_normal([0,0,0,0,0], cov=cov_mat, size = res_mat.shape[1])
        
        Tair_hat = Tair_hat + noise[:,0]
        Precip_hat = Precip_hat + noise[:,1]
        VPD_hat = VPD_hat + noise[:,2]
        PAR_hat = PAR_hat + noise[:,3]
        fapar_hat = fapar_hat + noise[:,4]
        
        preds = np.column_stack([X_new['year'], X_new['DOY'], Tair_hat, Precip_hat, VPD_hat, PAR_hat, fapar_hat])
        
        d.append(preds)

    climsims = pd.DataFrame(np.concatenate(d), columns = ['year', 'DOY', 'TAir', 'Precip', 'VPD', 'PAR', 'fAPAR'])
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

    d = []
    for i in range(x.shape[0]):
        out.loc[out['name'].isin(parameters), 'def'] = x[i,:]
        d.append(out['def'])

    d = np.array(d)

    return(d[:,:30])


def gen_simulations(n, data_dir = '~/physics_guided_nn/data/'):
    
    c = climate_simulations(n_simulations = n)
    #np.savetext('climate_simulations.csv', c.to_numpy(), delimiter=';')
    ct = torch.tensor(c.to_numpy(), dtype=torch.float64)
    p = parameter_samples(n_samples = n)
    #np.savetext('parameter_simulations.csv', p, delimiter=';')
    pt = torch.tensor(p, dtype=torch.float64)

    out = models.physical_forward(parameters = pt[0,:], input_data=ct)
    out = out.detach().numpy()
    #np.savetext('gpp_simulations.csv')
    
    c['GPP'] = out
    c.to_csv(''.join((data_dir, 'DA_preles_sims.csv')), index=False)

''' 
In case you want to make plots on the cluster computer, save them directly to results.

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(predsTair)
fig.savefig('results/temp.png')
''' 
