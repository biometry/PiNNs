# !/usr/bin/env python
# coding: utf-8
import pandas as pd
import sqlite3
import numpy as np
from scipy.optimize import leastsq
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


class ProfoundData:
    '''
    Extract necessary data from Profound DataBase and preprocess it
    '''
    def __init__(self, split, dir=None):
        self.split = split
        if dir:
            self.path_to_db = dir
        else:
            self.path_to_db = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/ProfoundData.sqlite'
        self.con = sqlite3.connect(self.path_to_db)


    def get_table(self, table_name, connection):
        df = pd.read_sql_query(''.join(('SELECT * FROM ', table_name)), connection)
        return df


    def get_clim(self, station_id):
        data = self.get_table('CLIMATEFLUXNET_master', self.con)
        data = data.loc[data['site_id'] == station_id, ['date', 'rad_Jcm2day', 'p_mm', 'tmean_degC']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1)
        return data


    def get_fapar(self, station_id):
        datafpar = self.get_table('MODIS', self.con)
        data = datafpar.loc[datafpar['site_id'] == station_id, ['date', 'fpar_percent']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1)
        data['fapar'] = data['fpar_percent'].copy()
        return data['fapar']


    def get_vpd(self, station_id):
        datavpd = self.get_table('METEOROLOGICAL', self.con)
        data = datavpd.loc[datavpd['site_id'] == station_id, ['date', 'year', 'mo', 'day', 'vpdFMDS_hPa']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['year', 'mo'], axis=1)
        data = data.resample('D').mean()
        data['VPD'] = data['vpdFMDS_hPa'].copy()
        return data['VPD'].copy()


    def get_gpp(self, station_id):
        datagpp = self.get_table('FLUX', self.con)
        data = datagpp.loc[datagpp['site_id'] == station_id, ['date', 'day', 'gppDtCutRef_umolCO2m2s1', 'gppDtCutSe_umolCO2m2s1']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1)
        data['GPP'] = data['gppDtCutRef_umolCO2m2s1'].values.copy() * 10 ** -6 * 0.012011 * 1000 * 86400
        data = data.resample('D').mean()
        return data['GPP'].copy()

    def merge_dat(self, d1, d2):
        out = d1.join(d2, how='outer')
        return out

    def normalize(self, var):
        z = (var - np.nanmean(var))/np.nanstd(var)
        return z

    def __getitem__(self):
        if self.split == 'training':
            self.sid = 12

        fapar = self.get_fapar(self.sid)
        clim = self.get_clim(self.sid)
        vpd = self.get_vpd(self.sid)
        gpp = self.get_gpp(self.sid)

        return fapar, clim, vpd, gpp



fapar, clim, vpd, gpp = ProfoundData('training').__getitem__()

plt.plot(gpp)
plt.show()



