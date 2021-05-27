

import torch
import torch.nn as nn
# ref: https://github.com/PhysicsNAS/PhysicsNAS/blob/master/collision/operations.py

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, inp):
        out = inp
        return out


class FCRelu(nn.Module):
    def __init__(self, dimension_in, dimension_out):
        super(FCRelu, self).__init__()
        self.op = nn.Sequential(
            nn.Linear(dimension_in, dimension_out),
            nn.ReLU(inplace=False)
        )


    def forward(self, inp):
        return self.op(inp)


class FCOut(nn.Module):
    def __init__(self, dimension_in, dimension_out):
        super(FCOut, self).__init__()
        self.op = nn.Sequential(
            nn.Linear(dimension_in, dimension_out)
        )

    def forward(self, inp):
        return self.op(inp)


# Physical Forward Pass
# class physical_forward(nn.Module): ...


# List operations
# add physical forward pass to first dict pair (not in final pass)
# hidden layers
operation_dict_different_dim = {'fc_relu': lambda dimension_in, dimension_out: FCRelu(dimension_in, dimension_out)}
operation_dict_similar_dim = {'fc_relu': lambda dimension_in, dimension_out: FCRelu(dimension_in, dimension_out),
                              'skip_connection': lambda dimension_in, dimension_out: Identity()}
# output layer
operation_dict_different_dim_out = {'fc_out': lambda dimension_in, dimension_out: FCOut(dimension_in, dimension_out)}
operation_dict_similar_dim_out = {'fc_out': lambda dimension_in, dimension_out: FCOut(dimension_in, dimension_out),
                                  'skip_connection': lambda dimension_in, dimension_out: Identity()}
