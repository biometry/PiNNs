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
    def __init__(self, split, handsoff=True, dir=None):
        self.split = split
        if dir:
            self.path_to_db = dir
        else:
            self.path_to_db = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/ProfoundData.sqlite'
        self.con = sqlite3.connect(self.path_to_db)
        self.handsoff = handsoff

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
        start = data.index[0]
        end = data.index[-1]
        data = data[data.index.isin(pd.date_range(start, end))]['fapar']
        idx = data[np.isnan(data)].index
        # mean bevor and after
        for i in idx:
            after = data.shift(periods=-1)
            bevor = data.shift(periods=1)
            if np.isfinite(bevor[i]) and np.isfinite(after[i]):
                data[i] = np.nanmean([bevor[i], after[i]])
            elif np.isfinite(bevor[i]):
                data[i] = bevor[i]
            elif np.isfinite(after[i]):
                data[i] = after[i]

        newdata = pd.DataFrame(data={'date': pd.date_range(start, end)})
        newdata = newdata.set_index(pd.to_datetime(newdata['date'])).drop(['date'], axis=1)
        fapar = newdata.join(data)
        nans = np.where(np.isnan(fapar.fapar))[0]
        # set value for days of resolution (usually 16 following days)
        for i in nans:
            fapar.iloc[i]['fapar'] = fapar.iloc[i - 1]['fapar']
        data = fapar
        return data

# calculate vpd
    def get_vpd(self, station_id):
        datavpd = self.get_table('METEOROLOGICAL', self.con)
        data = datavpd.loc[datavpd['site_id'] == station_id, ['date', 'vpdFMDS_hPa']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.resample('D').mean()
        data['VPD'] = data['vpdFMDS_hPa'].copy()/10 #hPa to kPa
        return data

    def get_et(self, station_id):
        dataet = self.get_table('ATMOSPHERICHEATCONDUCTION', self.con)
        data = dataet.loc[dataet['site_id'] == station_id, ['date', 'leCORR_Wm2']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.resample('D').mean()
        data['ET'] = (data.leCORR_Wm2 / 2257) * 0.001 * 86400
        # correct neg values
        pos = data.loc[data['ET'] < 0.0]
        bevor = data.shift(periods=1)
        after = data.shift(periods=-1)
        for p in pos.index:
            if bevor['ET'][p] >= 0 and after['ET'][p] >= 0:
                data['ET'][p] = np.mean([bevor['ET'][p], after['ET'][p]])
            elif bevor['ET'][p] >= 0:
                data['ET'][p] = bevor['ET'][p]
            elif after['ET'][p] >= 0:
                data['ET'][p] = after['ET'][p]
            else:
                data['ET'][p] = 0

        return data


    def get_gpp(self, station_id):
        datagpp = self.get_table('FLUX', self.con)
        data = datagpp.loc[datagpp['site_id'] == station_id, ['date', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1']]
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

    def shorten_merge(self, GPP, ET, Clim, VPD, fAPAR, lack = None, period=None):
        if not period:
            out = GPP.merge(ET, how='inner', on=['date']).merge(Clim, how='inner', on=['date']).merge(VPD, how='inner', on=['date']).merge(fAPAR, how='inner', on=['date'])
        return out



    def __getitem__(self):
        if self.split == 'validation':
            self.sid = 12 #hyytiala
        if self.split == 'training':
            self.sid = 21 # soro
        if self.split == 'test':
            self.sid = 5 #collelongo
        if self.split == 'NAS':
            self.sid = 14 #le bray

        fapar = self.get_fapar(self.sid)
        clim = self.get_clim(self.sid)
        vpd = self.get_vpd(self.sid)
        gpp = self.get_gpp(self.sid)
        et = self.get_et(self.sid)

        # merge data
        op = self.shorten_merge(gpp, et, clim, vpd, fapar)

        # shorten to time slot with all variables available
        if self.sid == 12:
            outix = pd.date_range('2004-01-01', '2005-12-31').union(pd.date_range('2007-01-01', '2012-12-31'))
        if self.sid == 21:
            outix = pd.date_range('2002-01-01', '2008-12-31')
        if self.sid == 5:
            outix = pd.date_range('1997-01-01', '2002-12-31').union(pd.date_range('2004-01-01', '2014-12-31'))
        if self.sid == 14:
            outix = pd.date_range('1997-01-01', '2001-12-31').union(pd.date_range('2003-01-01', '2008-12-31'))
        output = op[op.index.isin(outix)]

        # normalize
        if not self.handsoff:
            #normalize
            pass

        return output

hyytiala = ProfoundData('validation').__getitem__()





'''
# extract years containing all data
hyt = ProfoundData('validation').__getitem__()
hytindex = pd.date_range('2004-01-01', '2005-12-31').union(pd.date_range('2007-01-01', '2012-12-31'))
hyytiala = hyt[hyt.index.isin(hytindex)]

sor = ProfoundData('training').__getitem__()
soroindex = pd.date_range('2002-01-01', '2008-12-31')
soro = sor[sor.index.isin(soroindex)]

leb = ProfoundData('NAS').__getitem__()
lebindex = pd.date_range('1997-01-01', '2001-12-31').union(pd.date_range('2003-01-01', '2008-12-31'))
lebray = leb[leb.index.isin(lebindex)]

col = ProfoundData('test').__getitem__()
colindex = pd.date_range('1997-01-01', '2002-12-31').union(pd.date_range('2004-01-01', '2014-12-31'))
collelongo = col[col.index.isin(colindex)]
'''

