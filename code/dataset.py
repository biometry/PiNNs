# !/usr/bin/env python
# coding: utf-8
import pandas as pd
import sqlite3

path_to_db = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/ProfoundData.sqlite'
table = 'CLIMATEFLUXNET_master'



#class ProfoundData():


def connect_to_db(path):
    con = sqlite3.connect(path)
    return con


def get_table(table_name, connection):
    df = pd.read_sql_query(''.join(('SELECT * FROM ', table_name)), connection)
    return df


def get_clim(path, station_id):
    con = connect_to_db(path)
    data = get_table('CLIMATEFLUXNET_master', con)
    dat = data.loc[data['site_id'] == station_id, ['date', 'day', 'rad_Jcm2day', 'p_mm', 'tmean_degC']]
    return dat

def get_fapar(path, station_id):
    con = connect_to_db(path)
    datafpar = get_table('MODIS', con)
    dat = datafpar.loc[datafpar['site_id'] == station_id, ['date', 'day', 'fpar_percent']]
    return dat


def get_vpd(path, station_id):
    con = connect_to_db(path)
    datavpd = get_table('METEOROLOGICAL', con)
    dat = datavpd.loc[datavpd['site_id'] == station_id, ['date', 'year', 'mo', 'day', 'vpdFMDS_hPa']]
    datn = dat.set_index(pd.to_datetime(dat['date']))
    data = datn.drop(['year', 'mo'], axis=1)
    data = data.resample('D').mean()
    return data


def get_gpp(path, station_id):
    con = connect_to_db(path)
    datagpp = get_table('FLUX', con)
    dat = datagpp.loc[datagpp['site_id'] == station_id, ['date', 'day', 'neeVutRef_umolCO2m2s1']]
    return dat



f = get_gpp(path_to_db, 12)
f.info()
print(f)


fapar = get_fapar(path_to_db, 12)
clim = get_clim(path_to_db, 12)
vpd = get_vpd(path_to_db, 12)




b = pd.merge(fapar, clim, on=['date', 'day'], how='outer')
bb = b.set_index(pd.to_datetime(b['date']))
b = bb.drop(['date'], axis=1)

a = pd.merge(b, vpd, on=['date', 'day'], how='outer')
a.info()

#b = pd.merge(a, co2, on=['date', 'day'])

#co2.info()
#print(co2)