# !/usr/bin/env python
# coding: utf-8
import math
import numpy as np
import pandas as pd
import torch

def standardize(var, scaling=None):
    '''
    This function standardizes variables around the mean and the standard deviation
    :param var: two dimensional array of data points to normalize e.g. pd.DataFrame, torch.tensor
    :param scaling: other targets to normalize on
    :return: scaled variables in 2-D array
    '''
    if not scaling:
        if (isinstance(var, pd.DataFrame)):
            out = (var - np.mean(var)) / np.std(var)
        elif (torch.is_tensor(var)):
            out = (var - torch.mean(var)) / torch.std(var)
        else:
            out = (var - np.mean(var, axis=0)) / np.std(var, axis=0)
    else:
        out = (var - scaling[0]) / scaling[1]

    return out


def encode_doy(doy):
    '''
    Encode cyclic feature as doy [1, 365] or [1, 366]
    :param doy: cyclic feature e.g. doy [1, 365]
    :return: encoded feature in sine and cosine
    '''
    normalized_doy = doy * (2. * np.pi / 365)
    return np.sin(normalized_doy), np.cos(normalized_doy)

