# !/usr/bin/env python
# coding: utf-8
from PNAS import *
from sklearn import metrics
import utils
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.model_selection import train_test_split

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)


# define options
options = {
    'num_epochs': 5000,
    'lr': 0.015,
    'arch_lr': 0.015,
    'weight_decay': 0,
    'arch_weight_decay': 1e-3,
    'gradient_clip': 10,
    'batch_size': 4,
    'criterion': metrics.mean_absolute_error,
    'figure_dir': './',
    'network_dir': './'
}

x, y = utils.loaddata('NAS', 1, dir='./data/')

x_train, x_test, y_train, y_test = train_test_split(x, y)

train_set = TensorDataset(Tensor(x_train.to_numpy()), Tensor(y_train.to_numpy()))
test_set = TensorDataset(Tensor(x_test.to_numpy()), Tensor(y_test.to_numpy()))

model = SearchController(
    dim_in = x.shape[1],
    dim_out= y.shape[1],
    nlayers=5,
    dim_hid=128,
    criterion=options['criterion']
)

print(model.get_cur_genotype())
best_genotype = search_arch(model, train_set, test_set, options)
