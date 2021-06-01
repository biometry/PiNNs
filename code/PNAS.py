# !/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

from operationsPNAS import *
from plot import *

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
        self.init_input_dim = [init_input_dim]
        self.op_name_list = []
        self.n_init_inp = len([init_input_dim])

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
        for in_dim in self.init_input_dim:
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
        self.nini_inp = [self.ini_inp_dims]

        # Build Network
        self.network = Network(self.nlayers, self.ini_inp_dims, self.dim_hid, self.dim_out)
        # Operation List
        self.op_name_list = self.network.get_op_name_list()

        # Alpha for operations
        self.op_alpha_list = []
        for i in range(len(self.op_name_list)):
            self.op_alpha_list.append(
                torch.randn(len(self.op_name_list[i]), requires_grad=True, device=device)
            )
        # Alpha for edges
        self.edge_alpha_list = []
        for i in range(self.nlayers):
            self.edge_alpha_list.append(
                torch.randn(i + len(self.nini_inp), requires_grad=True, device=device)
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
        n = len(self.nini_inp)
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


def search_arch(model, train_set, test_set, options):
    print('Train:', len(train_set))
    print('Test:', len(test_set))
    print(train_set)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set))
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=len(test_set))
    train_set_size = len(train_set)
    sample_id = list(range(len(train_set)))
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_id[:train_set_size//2])
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_id[train_set_size // 2:])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=options['batch_size'], sampler= train_sampler, shuffle=False)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=options['batch_size'], sampler= val_sampler, shuffle=False)
    print(train_loader)
    print(val_loader)
    # Optimizers
    alpha_optimizer = torch.optim.Adam(model.arch_parameters(), lr=options['arch_lr'], weight_decay=options['arch_weight_decay'])
    weight_optimizer = torch.optim.Adam(model.parameters(), lr= options['lr'], weight_decay=options['weight_decay'])

    weight_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        weight_optimizer, float(options['num_epochs']), eta_min= 1e-3 * options['lr']
    )
    alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        alpha_optimizer, float(options['num_epochs']), eta_min= 1e-3 * options['arch_lr']
    )
    # Best error
    min_test_diff = float('inf')
    epoch_loss = []

    # Best genotype
    best_genotype = None

    # Train loop
    for epoch in range(options['num_epochs']):

        # Train
        train_stat = train_model(model,
                                 train_loader,
                                 val_loader,
                                 alpha_optimizer,
                                 weight_optimizer,
                                 options['criterion'],
                                 options=None
                                 )

        # Test
        test_stat = test_model(model, test_loader, options['criterion'])
        cur_test_diff = test_stat['diff']

        # Update Scheduler
        weight_scheduler.step()
        alpha_scheduler.step()

        # Output information
        cur_genotype = model.get_cur_genotype()
        print('-' * 20 + 'Epoch ' + str(epoch) + '-' * 20)
        print('Train:\t', train_stat)
        print('Test:\t', test_stat)

        # Save
        if min_test_diff > cur_test_diff:

            min_test_diff = cur_test_diff
            best_genotype = cur_genotype
            print('Best')
            print(best_genotype)

            # Save the network
            if options['network_dir']:
                torch.save(model.state_dict(), options['network_dir'])

            # Plot the architecture picture
            if options['figure_dir']:
                plot_genotype(
                    cur_genotype,
                    file_name='test_' + str(epoch),
                    figure_dir=options['figure_dir'],
                    save_figure=True
                )
                print('Figure saved.')
    return best_genotype


def train_model(model, train_loader, val_loader, alpha_optimizer, weight_optimizer, criterion, options):
    model.train()

    mse_distance = nn.MSELoss()

    batch_loss = []
    batch_diff = []
    for step, (train_sample, val_sample) in enumerate(zip(train_loader, val_loader)):
        # Train set
        print(train_sample)
        print(val_sample)
        x_train = train_sample[0]
        y_train = train_sample[1]

        # Val set
        x_val = val_sample[0]
        y_val = val_sample[1]

        alpha_optimizer.zero_grad()
        y_hat_val = model(x_val)
        loss = criterion(y_hat_val, y_val)
        loss.backward()

        # Update weights
        alpha_optimizer.step()

        # Network step
        weight_optimizer.zero_grad()
        y_hat_train = model(x_train)
        loss = criterion(y_hat_train, y_train)
        loss.backward()

        # Clip the gradient
        nn.utils.clip_grad_norm_(model.parameters(), options['gradient_clip'])

        # Update weights
        weight_optimizer.step()

        # Train statistics
        batch_loss.append(loss.item())
        batch_diff.append(mse_distance(y_hat_train, y_train).item())

    train_loss = sum(batch_loss) / len(batch_loss)
    train_diff = sum(batch_diff) / len(batch_diff)

    return {'loss': train_loss, 'diff': train_diff}


def test_model(model, test_loader, criterion):
    model.eval()
    mse_distance = nn.MSELoss()
    batch_loss = []
    batch_diff = []
    with torch.no_grad():
        for step, test_sample in enumerate(test_loader):
            # Test set
            x_test = test_sample[0]
            y_test = test_sample[1]

            # Network output
            y_hat_test = model(x_test)
            loss = criterion(y_hat_test, y_test)

            # Test statistics
            batch_loss.append(loss.item())
            batch_diff.append(mse_distance(y_hat_test, y_test).item())

    test_loss = sum(batch_loss) / len(batch_loss)
    test_diff = sum(batch_diff) / len(batch_diff)

    return {'loss': test_loss, 'diff': test_diff}
