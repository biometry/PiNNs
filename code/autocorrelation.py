# !/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

data_path = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/'
soro = pd.read_csv(''.join((data_path, 'soro.csv')))
hyytiala = pd.read_csv(''.join((data_path, 'hyytiala.csv')))
collelongo = pd.read_csv(''.join((data_path, 'collelongo.csv')))

plot_pacf(collelongo['Tair'], lags=30)
plot_pacf(soro['Tair'], lags=30)
plot_pacf(hyytiala['Tair'], lags=30)
plt.show()
