# !/usr/bin/env python
# coding: utf-8
import pandas as pd
import sqlite3
import numpy as np
import zipfile
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
        if dir.endswith('.zip'):
            zf = zipfile.ZipFile(dir)
            db = zf.read('ProfoundData.sqlite')
            self.path_to_db = db
        elif dir.endswith('.sqlite'):
            self.path_to_db = dir
        else:
            self.path_to_db = '' # Path to sqlite file
        self.con = sqlite3.connect(self.path_to_db)
        self.handsoff = handsoff

    def get_table(self, table_name, connection):
        df = pd.read_sql_query(''.join(('SELECT * FROM ', table_name)), connection)
        return df


    def get_clim(self, station_id):
        data = self.get_table('CLIMATEFLUXNET_master', self.con)
        data = data.loc[data['site_id'] == station_id, ['date', 'rad_Jcm2day', 'p_mm', 'tmean_degC']]
        data = data.set_index(pd.to_datetime(data['date']))
        data['Precip'] = data['p_mm'].copy()
        data['Tair'] = data['tmean_degC'].copy()
        # global irradiance to PAR
        # reference: Taiz & Zeiger (2015): Plant Physiology and Development, Ch.9 (p.172-174)
        data['PAR'] = ((data.rad_Jcm2day * (2.2*(10**(-7))) / (299792458 * (6.626070150 * (10 **(-34)))))/(6.02*(10**23))) / 0.0001
        data = data.drop(['date', 'p_mm', 'tmean_degC', 'rad_Jcm2day'], axis=1)
        return data


    def get_fapar(self, station_id):
        datafpar = self.get_table('MODIS', self.con)
        data = datafpar.loc[datafpar['site_id'] == station_id, ['date', 'fpar_percent']]
        data = data.set_index(pd.to_datetime(data['date']))
        data['fapar'] = data['fpar_percent'].copy()
        data = data.drop(['date', 'fpar_percent'], axis=1)
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


    def get_vpd(self, station_id):
        datavpd = self.get_table('METEOROLOGICAL', self.con)
        data = datavpd.loc[datavpd['site_id'] == station_id, ['date', 'vpdFMDS_hPa']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.resample('D').mean()
        data['VPD'] = data['vpdFMDS_hPa'].copy()/10 #hPa to kPa
        data = data.drop(['vpdFMDS_hPa'], axis=1)
        return data

    def get_et(self, station_id):
        dataet = self.get_table('ATMOSPHERICHEATCONDUCTION', self.con)
        data = dataet.loc[dataet['site_id'] == station_id, ['date', 'leCORR_Wm2']]
        data = data.set_index(pd.to_datetime(data['date']))
        data = data.resample('D').mean()
        data['LE'] = data['leCORR_Wm2'].copy()
        data = data.drop(['leCORR_Wm2'], axis=1)
        # correct neg values
        pos = data.loc[data['LE'] < 0.0]
        bevor = data.shift(periods=1)
        after = data.shift(periods=-1)
        for p in pos.index:
            if bevor['LE'][p] >= 0 and after['LE'][p] >= 0:
                data['LE'][p] = np.mean([bevor['LE'][p], after['LE'][p]])
            elif bevor['LE'][p] >= 0:
                data['LE'][p] = bevor['LE'][p]
            elif after['LE'][p] >= 0:
                data['LE'][p] = after['LE'][p]
            else:
                data['LE'][p] = 0
        return data


    def get_gpp(self, station_id):
        datagpp = self.get_table('FLUX', self.con)
        data = datagpp.loc[datagpp['site_id'] == station_id, ['date', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1']]
        data = data.set_index(pd.to_datetime(data['date']))
        data['GPP'] = data['gppDtVutRef_umolCO2m2s1'].values.copy() * 10 ** -6 * 0.012011 * 1000 * 86400
        data = data.drop(['date', 'gppDtVutRef_umolCO2m2s1', 'gppDtVutSe_umolCO2m2s1'], axis=1)
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


    def shorten_merge(self, GPP, ET, Clim, VPD, fAPAR):
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
            outix = pd.date_range('2001-01-01', '2002-12-31').union(pd.date_range('2004-01-01', '2014-12-31'))
        if self.sid == 14:
            outix = pd.date_range('2001-01-01', '2001-12-31').union(pd.date_range('2003-01-01', '2008-12-31'))
        output = op


        # latent heat to ET
        # latent heat vaporization ref: Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
        #                                                Kluwer Academic Publishers, Dordrecht, Netherlands
        tair = output['Tair']
        LE = output['LE']
        output['ET'] = np.array([0]*len(output['LE']))
        k1 = 2.501
        k2 = 0.00237
        lb = (k1 - k2 * tair) * 1e06
        output['ET'] = (LE / lb) * 86400
        output = output.drop(['LE'], axis=1)
        output['DOY'] = output.index.dayofyear
        output = output[output.index.isin(outix)]
        output['CO2'] = np.array([380]*len(output.index))

        return output


#ProfoundData('NAS', dir='C:/Users/Niklas/Downloads/ProfoundData.zip')




'''
hyytiala = ProfoundData('validation').__getitem__()
soro = ProfoundData('training').__getitem__()
collelongo = ProfoundData('test').__getitem__()
lebray = ProfoundData('NAS').__getitem__()


data_path = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/'
hyytiala.to_csv(''.join((data_path, 'hyytiala.csv')))
soro.to_csv(''.join((data_path, 'soro.csv')))
collelongo.to_csv(''.join((data_path, 'collelongo.csv')))
lebray.to_csv(''.join((data_path, 'lebray.csv')))
'''
