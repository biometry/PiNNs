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

# implement conversion of units
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
        return data

# calculate vpd
    def get_vpd(self, station_id):
        datavpd = self.get_table('METEOROLOGICAL', self.con)
        data = datavpd.loc[datavpd['site_id'] == station_id, ['date', 'year', 'mo', 'day', 'vpdFMDS_hPa']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['year', 'mo'], axis=1)
        data = data.resample('D').mean()
        data['VPD'] = data['vpdFMDS_hPa'].copy()/10 #hPa to kPa
        return data


    def get_gpp(self, station_id):
        datagpp = self.get_table('FLUX', self.con)
        data = datagpp.loc[datagpp['site_id'] == station_id, ['date', 'day', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1)
        data['GPP'] = data['gppDtVutRef_umolCO2m2s1'].values.copy() * 10 ** -6 * 0.012011 * 1000 * 86400
        data = data.resample('D').mean()
        return data

    def get_swc(self, station_id):
        dataswc = self.get_table('SOILTS', self.con)
        data = dataswc.loc[dataswc['site_id'] == station_id, ['date', 'swcFMDS1_degC']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.drop(['date'], axis=1)
        data['SWC'] = data['swcFMDS1_degC'].values.copy()
        data = data.resample('D').mean()
        return data


    def merge_dat(self, d1, d2):
        out = d1.join(d2, how='outer')
        return out

    def normalize(self, var):
        z = (var - np.nanmean(var))/np.nanstd(var)
        return z

    def shorten_merge(self, GPP, Clim, VPD, fAPAR=None, lack = None, period=None):
        if not period:
            out = GPP.merge(Clim, left_index=True).merge(VPD, left_index=True)
        return out


        '''
        if not period:
            period = ["2000-01-01", "2012-12-31"]# shorten dataset on gpp availables
        out = data[period[0]:period[1]]
        if lack:
            pass #remove data of lack years
        return out
        '''
# where to get ET??


    def __getitem__(self):
        if self.split == 'validation':
            self.sid = 12 #hyytiala
        if self.split == 'training':
            self.sid = 3 #bily kriz
        if self.split == 'test':
            self.sid = 5 #collelongo
        if self.split == 'NAS':
            self.sid = 14 #le bray

        #fapar = self.get_fapar(self.sid)
        clim = self.get_clim(self.sid)
        vpd = self.get_vpd(self.sid)
        gpp = self.get_gpp(self.sid)
        #swc = self.get_swc(self.sid)

        output = self.shorten_merge(gpp, clim, vpd)

        return output


op = ProfoundData('validation').__getitem__()
print(op)
print(op.info())


'''
data1 = ProfoundData('test').__getitem__()
data2 = ProfoundData('training').__getitem__()
data3 = ProfoundData('NAS').__getitem__()
data4 = ProfoundData('validation').__getitem__()
plt.plot(data1, label='collelongo')
plt.plot(data2, label='bily kriz')
plt.plot(data3, label='le bray')
plt.plot(data4, label='hyytiala')
plt.show()

gpp2 = ProfoundData('test').__getitem__()
gpp3 = ProfoundData('validation').__getitem__()
gpp4 = ProfoundData('NAS').__getitem__()

plt.plot(gpp1, label='bily kriz')
plt.plot(gpp2, label='collelongo')
plt.plot(gpp3, label='hyytiala')
plt.plot(gpp4, label='le bray')
plt.legend()
plt.show()
'''


