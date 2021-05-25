# !/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_pacf

data_path = 'C:/Users/Niklas/Desktop/Uni/M.Sc. Environmental Science/Thesis/physics_guided_nn/data/'
data = pd.read_csv(''.join((data_path, 'soro.csv')))

plot_pacf(data['Precip'])
plt.show()