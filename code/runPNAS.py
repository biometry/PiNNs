# !/usr/bin/env python
# coding: utf-8
from PNAS import *
from sklearn import metrics
import utils
from sklearn.model_selection import train_test_split

# define options
options = {
    'num_epochs': 5000,
    'lr': 0.015,
    'arch_lr': 0.015,
    'weight_decay': 0,
    'arch_weight_decay': 1e-3,
    'gradient_clip': 10,
    'batch_size': 4,
    'criterion': metrics.mean_absolute_error(),
    'figure_dir': './',
    'network_dir': './'
}

x, y = utils.loaddata('NAS', 1, dir='./data')

x_train, x_test, y_train, y_test = train_test_split(x, y)

train_set = {'X': x_train, 'Y': y_train}
test_set = {'X': x_test, 'Y': y_test}

model = SearchController(
    dim_in = x.shape[1],
    dim_out=y.shape[1],
    nlayers=5,
    dim_hid=128,
    criterion=options['criterion']
)

print(model.get_cur_genotype())
best_genotype = search_arch(model, train_set, test_set, options)
