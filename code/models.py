# !/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn

# naive feed forward MLP
class NMLP(nn.Module):

    def __init__(self, input_dim, output_dim, model_design):
        super(NMLP, self).__init__()

        self.MLP = nn.Sequential()
        self.nlayers = len(model_design['layer_sizes'])

        for i in range(1, self.nlayers):
            if i == 1:
                self.MLP.add_module('Linear', nn.Linear(input_dim, model_design['layer_sizes'][i]))
                self.MLP.add_module('relu1', nn.ReLU(inplace=True))
            if i == self.nlayers-1:
                self.MLP.add_module('Linear', nn.Linear(model_design['layer_sizes'][i-1], output_dim))
            else:
                self.MLP.add_module('Linear', nn.Linear(model_design['layer_sizes'][i-1], model_design['layer_sizes'][i]))
                self.MLP.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.MLP(x)

        return out







