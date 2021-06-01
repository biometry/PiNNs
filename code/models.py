# !/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn

# naive feed forward MLP
class NMLP(nn.Module):

    def __init__(self, input_dim, output_dim, layersizes):
        super(NMLP, self).__init__()

        self.layers = nn.Sequential()
        self.nlayers = len(layersizes)
        print('Initializing model')
        for i in range(0, self.nlayers+1):
            if i == 0:
                self.layers.add_module(f'input{i}', nn.Linear(input_dim, layersizes[i]))
                self.layers.add_module(f'activation{i}', nn.ReLU())
                print('adding input l', input_dim, layersizes[i])
            elif i == (self.nlayers):
                self.layers.add_module(f'output{i}', nn.Linear(layersizes[i-1], output_dim))
                print('adding output l', layersizes[i-1], output_dim)
            elif i and i < self.nlayers:
                self.layers.add_module(f'fc{i}', nn.Linear(layersizes[i-1], layersizes[i]))
                self.layers.add_module(f'activation{i}', nn.ReLU())
                print('adding hidden l', layersizes[i-1], layersizes[i])
                

    def forward(self, x):
        out = self.layers(x)

        return out







