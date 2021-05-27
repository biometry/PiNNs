# !/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from operationsPNAS import *

# Create Layer of all operations
class MixedLayer(nn.Module):
    def __init__(self, dim_in, dim_out, op_dict):
        super(MixedLayer, self).__init__()

        self.layers = nn.ModuleList()
        for operation in op_dict.keys():
            layer = op_dict[operation](dim_in, dim_out)
            self.layers.append(layer)

    def forward(self, x, weights):
        out = [w *  layer(x) for w, layer in zip(weights, self.layers)]
        return sum(out)


class Network(nn.Module):
    def __init__(self, nlayers, init_input_dim, dim_hid, dim_out):
        super(Network, self).__init__()

        self.nlayers = nlayers
        self.dim_hid = dim_hid
        self.dim_out = dim_out
        self.layers = nn.ModuleList()
        self.init_input_dim = init_input_dim
        self.op_name_list = []
        self.n_init_inp = len(init_input_dim)

        for i in range(self.nlayers-1):
            for j in range(i + self.n_init_inp):

                # input layer
                if j < self.n_init_inp:
                    layer = MixedLayer(self.init_input_dim[j], self.dim_hid, operation_dict_different_dim)
                    self.layers.append(layer)
                    self.op_name_list.append(list(operation_dict_different_dim.keys()))

                # layers in between
                else:
                    layer = MixedLayer(self.dim_hid, self.dim_hid, operation_dict_similar_dim)
                    self.layers.append(layer)
                    self.op_name_list.append(list(operation_dict_similar_dim.keys()))
        # Last Layer
        # from input -> output
        for in_dim in init_input_dim:
            if in_dim == self.dim_out:
                layer = MixedLayer(in_dim, self.dim_out, operation_dict_similar_dim_out)
                self.layers.append(layer)
                self.op_name_list.append(list(operation_dict_similar_dim_out.keys()))
            else:
                layer = MixedLayer(in_dim, self.dim_out, operation_dict_different_dim_out)
                self.layers.append(layer)
                self.op_name_list.append(list(operation_dict_similar_dim_out.keys()))
        # from hidden -> output
        for j in range(self.nlayers-1):
            layer = MixedLayer(self.dim_hid, self.dim_out, operation_dict_different_dim_out)
            self.layers.append(layer)
            self.op_name_list.append(list(operation_dict_different_dim_out.keys()))

    def forward(self, s0, w_edge, w_op):
        states = [s0]
        off = 0

        # Inp all previous layers
        for i in range(self.nlayers):
            s = sum(w_edge[i][j] * self.layers[off + j](cur_state, w_op[off + j]) for j, cur_state in enumerate(states))
            off += len(states)
            states.append(s)

        return states[-1]

    def get_op_name_list(self):
        return self.op_name_list


# Search Controller
class SearchController(nn.Module):
    def __init__(self, dim_in, dim_out, nlayers, dim_hid, criterion):
        super(SearchController, self).__init__()
        self.nlayers = nlayers
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hid = dim_hid
        self.criterion = criterion
        self.ini_inp_dims = dim_in #depending on input of NN
        self.nini_inp = len(self.ini_inp_dims)

        # Build Network
        self.network = Network(self.nlayers, self.ini_inp_dims, self.dim_hid, self.dim_out)
        # Operation List
        self.op_name_list = self.network.get_op_name_list()

        # Alpha for operations
        self.op_alpha_list = []
        for i in range(len(self.op_name_list)):
            self.op_alpha_list.append(
                torch.randn(len(self.op_name_list[i]), requires_grad=True)
            )
        # Alpha for edges
        self.edge_alpha_list = []
        for i in range(len(self.nlayers)):
            self.edge_alpha_list.append(
                torch.randn(i + self.nini_inp, requires_grad=True)
            )
        # Ini alphas
        with torch.no_grad():
            for a in self.edge_alpha_list:
                a.mul_(1e-1)
        with torch.no_grad():
            for a in self.op_alpha_list:
                a.mul_(1e-1)

    def forward(self, x):
        y_hat = self.network(x, self.edge_weights_masked(), self.operation_weights_masked())
        return y_hat

    def arch_parameters(self):
        return self.op_alpha_list + self.edge_alpha_list

    def edge_weights(self):
        return [F.softmax(a, dim=-1) for a in self.edge_alpha_list]

    def operation_weights(self):
        return [F.softmax(a, dim=-1) for a in self.op_alpha_list]

    def edge_weights_masked(self):
        if self.training:
            return [RandomMask.apply(w, 2) for w in self.edge_weights()]
        else:
            mask_list = []
            for w in self.edge_weights():
                max_idx = torch.argsort(w, descending=True)[:2]
                mask = torch.zeros_like(w)
                mask[max_idx] = 1.0
                mask_list.append(mask)
            return mask_list

    def operation_weights_masked(self):
        if self.training:
            return [RandomMask.apply(w, 1) for w in self.operation_weights()]
        else:
            mask_list = []
            for weight_idx, weight in enumerate(self.operation_weights()):
                sorted_idxs = torch.argsort(weight, descending=True)
                max_idx = sorted_idxs[0]
                mask = torch.zeros_like(weight)
                mask[max_idx] = 1.0
                mask_list.append(mask)

            return mask_list

    def get_cur_genotype(self):
        edge_weights = [weight.data.cpu().numpy() for weight in self.edge_weights()]
        operation_weights = [weight.data.cpu().numpy() for weight in self.operation_weights()]
        gene = []
        n = self.nini_inp
        start = 0
        for i in range(self.nlayers):  # for each node
            end = start + n
            W = operation_weights[start:end].copy()

            best_edge_idx = np.argsort(edge_weights[i])[::-1]  # descending order

            for j in best_edge_idx[:2]:  # pick two best
                k_best = None
                for k in range(len(W[j])):  # get strongest ops for j->i
                    if k_best is None or W[j][k] > W[j][k_best]:
                        k_best = k
                gene.append((self.op_name_list[start + j][k_best], j))  # save ops and node
            start = end
            n += 1
        return gene


class RandomMask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w, nmasked_w):
        ctx.nmasked_w = nmasked_w
        picked_idx = torch.multinomial(w, nmasked_w)
        masked_w = torch.zeros_like(w)
        masked_w[picked_idx] = 1.0

        return masked_w

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out.clone(), None



