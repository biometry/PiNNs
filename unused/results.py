#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:47:12 2022
@author: Marieke_Wesselkamp
"""

import os

os.chdir("/")

import pandas as pd
import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# %%
def plot_performance(perf, prediction, data_use, log=False):
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Fira Sans OT']})

    fig = plt.figure(figsize=(7, 7), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    # Creating plot
    if log:
        bp = ax.boxplot(np.log(perf), patch_artist=True)
    else:
        bp = ax.boxplot(perf, patch_artist=True)

    colors = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6', '#998ec3']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='black',
                    linewidth=2.5,
                    linestyle=":")
    for median in bp['medians']:
        median.set(color='black',
                   linewidth=2)
    # if data_use == 'sparse':
    ax.set_xticklabels(
        ['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrized',
         'Domain\nAdaptation'])
    # else:
    #    ax.set_xticklabels(['', '', '', '', '', ''])

    if not log:
        if (prediction == 'temp') & (data_use == 'full'):
            ax.set_ylim(0.5, 2.25)
        if (prediction == 'temp') & (data_use == 'sparse'):
            ax.set_ylim(0.5, 4.5)
        if (prediction == 'spat') & (data_use == 'full'):
            ax.set_ylim(1.5, 4.0)
        if (prediction == 'spat') & (data_use == 'sparse'):
            ax.set_ylim(1.5, 4.0)

    if log:
        ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=20)
    else:
        ax.set_ylabel('Mean absolute error [g C m$^{-2}$ day$^{-1}$]', fontsize=20)

    if prediction == 'temp':
        pass
        # ax.set_title(f'Temporal prediction: {data_use} data', fontsize = 28, y=1.08)
    else:
        pass
        # ax.set_title(f'Spatial prediction: {data_use} data', fontsize = 28, y=1.08)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
        # tick.label.set_fontfamily({'font.serif':'Palatino'})
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        # tick.label.set_fontfamily('Palatino Linotype')

    plt.show()

    if log:
        fig.savefig(f'../plots/performance_{prediction}_{data_use}_log.pdf', bbox_inches='tight', dpi=200, format='pdf')
    else:
        fig.savefig(f'../plots/performance_{prediction}_{data_use}.pdf', bbox_inches='tight', dpi=200, format='pdf')


# %%
mods = ['preles', 'mlp', 'res', 'res2', 'reg', 'mlpDA3']
data_use = ['full', 'sparse']

perf = np.zeros((4, len(mods)))
perf2 = np.zeros((5, len(mods)))

fuckyoulist = []

for d in data_use:
    i = 0
    for mod in mods:

        if mod != 'preles':
            perf[:, i] = pd.read_csv(f"../resultsN2/{mod}_eval_{d}_performance.csv", index_col=False).iloc[:, 4]
            perf2[:, i] = pd.read_csv(f"../resultsN2/2{mod}_eval_{d}_performance.csv", index_col=False).iloc[:, 4]
        else:
            perf[:, i] = pd.read_csv(f"../resultsN2/{mod}_eval_{d}_performance.csv", index_col=False).iloc[:, 2]
            perf2[:, i] = pd.read_csv(f"../resultsN2/2{mod}_eval_{d}_performance.csv", index_col=False).iloc[:, 2]
            # pred = pd.read_csv(f"../resultsN/{mod}_{data_use}_eval_preds_test.csv",index_col=False )#.iloc[:,1:]
            # tloss = pd.read_csv(f"../resultsN/{mod}_trainloss_{data_use}.csv",index_col=False )#.iloc[:,1:]
            # vloss = pd.read_csv(f"../resultsN/{mod}_vloss_{data_use}.csv",index_col=False )#.iloc[:,1:]

        i += 1
    fuckyoulist.append(perf.copy())
    fuckyoulist.append(perf2.copy())

    plot_performance(perf, 'temp', d)
    plot_performance(perf2, 'spat', d)
    plot_performance(perf, 'temp', d, log=True)
    plot_performance(perf2, 'spat', d, log=True)


# %%
def plot_performance_summarized(dat1, dat2, prediction):
    plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Fira Sans OT']})

    def draw_plot(data, offset, edgecolor, colors):
        pos = np.arange(data.shape[1]) + offset
        bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True)

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            # whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='black',
                        linewidth=2.5,
                        linestyle=":")
        for median in bp['medians']:
            median.set(color='black',
                       linewidth=2)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=200)
    # ax = fig.add_axes([0,0,1,1])
    if prediction == 'temp':
        colors1 = ['#2c7bb6', '#2c7bb6', '#2c7bb6', '#2c7bb6', '#2c7bb6']
        colors2 = ['#abd9e9', '#abd9e9', '#abd9e9', '#abd9e9', '#abd9e9']
        red_patch = mpatches.Patch(color='#2c7bb6', label='Full')
        blue_patch = mpatches.Patch(color='#abd9e9', label='Sparse')
    else:
        colors1 = ['#fdae61', '#fdae61', '#fdae61', '#fdae61', '#fdae61']
        colors2 = ['#d7191c', '#d7191c', '#d7191c', '#d7191c', '#d7191c']
        red_patch = mpatches.Patch(color='#fdae61', label='Full')
        blue_patch = mpatches.Patch(color='#d7191c', label='Sparse')

    draw_plot(dat1, -0.2, edgecolor="black", colors=colors1)
    draw_plot(dat2, +0.2, edgecolor="black", colors=colors2)

    plt.xticks(np.arange(5))
    ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics',
                        'Regula-\nrized'])  # , 'Domain\nAdaptation'])
    ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=20)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)
        # tick.label.set_fontfamily({'font.serif':'Palatino'})
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        # tick.label.set_fontfamily('Palatino Linotype')
    # ax = fig.add_axes([0,0,1,1])
    if prediction == 'temp':
        plt.legend(handles=[red_patch, blue_patch], fontsize=18, loc="upper right")
    else:
        plt.legend(handles=[red_patch, blue_patch], fontsize=18, loc="lower left")

    plt.show()
    fig.savefig(f'../plots/performance_{prediction}_summarized.png', bbox_inches='tight', dpi=200)


# %%
plot_performance_summarized(np.log(fuckyoulist[0]), np.log(fuckyoulist[2]), prediction='temp')
plot_performance_summarized(np.log(fuckyoulist[1]), np.log(fuckyoulist[3]), prediction='spat')


# %%
def plot_via(vi, labels, var, vi2=None, vi3=None, main=None, xticks=None, xlabel=None, legend=True):
    vm = np.mean(vi.iloc[:, 1:], axis=1)
    vu = vm + 2 * np.std(vi.iloc[:, 1:], axis=1)
    vl = vm - 2 * np.std(vi.iloc[:, 1:], axis=1)

    fig, ax = plt.subplots(figsize=(7, 7))
    # ax.set_ylim((,10))
    ax.fill_between(np.arange(len(vm)), vu, vl, color='#fdae61', alpha=0.5, label=labels[0])
    ax.plot(vm, color="black")

    if not vi2 is None:
        vm = np.mean(vi2.iloc[:, 1:], axis=1)
        vu = vm + 2 * np.std(vi2.iloc[:, 1:], axis=1)
        vl = vm - 2 * np.std(vi2.iloc[:, 1:], axis=1)

        ax.fill_between(np.arange(len(vm)), vu, vl, color='#2c7bb6', alpha=0.5, label=labels[1])
        ax.plot(vm, color="black")

    if not vi3 is None:
        vm = np.mean(vi3.iloc[:, 1:], axis=1)
        vu = vm + 2 * np.std(vi3.iloc[:, 1:], axis=1)
        vl = vm - 2 * np.std(vi3.iloc[:, 1:], axis=1)

        ax.fill_between(np.arange(len(vm)), vu, vl, color='#abd9e9', alpha=0.5, label=labels[2])
        ax.plot(vm, color="black")

    if not xticks is None:
        if var == 'Tair':
            ax.xaxis.set_ticks([0, 33, 66, 100, 133, 166, 200])
        elif var == 'PAR':
            ax.xaxis.set_ticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
        elif var == 'Precip':
            ax.xaxis.set_ticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200])
        ax.set_xticklabels(xticks)

    if not main is None:
        ax.set_title(main, fontsize=24, y=1.08)

    for tick in ax.yaxis.get_major_ticks():
        print(tick)
        tick.label.set_fontsize(18)
        # tick.label.set_fontfamily({'font.serif':'Palatino'})
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)

    if legend:
        ax.legend(fontsize=18, loc=(0.7, 1.02))
    ax.set_ylabel("Predicted yearly GPP [g C m$^{-2}$ day$^{-1}$]", fontsize=18)

    if not xlabel is None:
        ax.set_xlabel(xlabel, fontsize=18)

    if legend:
        fig.savefig(f'../plots/VIA_{var}.pdf', bbox_inches='tight', dpi=300, format='pdf')
    else:
        fig.savefig(f'../plots/VIA_{var}2.pdf', bbox_inches='tight', dpi=300)


# %%
mods = ['preles', 'mlp', 'res', 'res2', 'reg']
data_use = ['full']  # , 'sparse']
variables = ['VPD', 'Precip', 'Tair', 'fapar', 'PAR']
xlabels = ['Vapor pressure deficit [kPA]', 'Precipiation [mm]', 'Mean air temperature [$^{\circ}$C]',
           'Fraction of photosynthetic active radiation', 'Photosynthetic active radiation [mol m$^{-2}$ d$^{-1}$]']

var = 'Tair'
mod = 'reg'
d = 'full'

i = 0
for var in variables:
    for d in data_use:
        vi = pd.read_csv(f"../resultsN2/mlp_{d}_{var}_via.csv", index_col=False, header=1)  #
        vi1 = pd.read_csv(f"../resultsN2/reg_{d}_{var}_via.csv", index_col=False, header=1)  # .iloc[:,1:]
        vi2 = pd.read_csv(f"../resultsN2/res2_{d}_{var}_via.csv", index_col=False, header=1)  # .iloc[:,1:]
        if var == 'fapar':
            xticks = [0] + list(np.round(vi.iloc[0::24, 0], 1))
        elif var == 'Tair':
            xticks = [-20, -10, 0, 10, 20, 30, 40]
        elif var == 'PAR':
            xticks = [0, 25, 50, 75, 100, 125, 150, 175, 200]
        elif var == 'Precip':
            xticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        else:
            xticks = [0] + [int(x) for x in list(np.rint(vi.iloc[0::25, 0]))] + [int(vi.iloc[-1, 0])]

        plot_via(vi, labels=['Naive', 'Regularized', 'Parallel Physics'],
                 var=var,
                 vi2=vi1, vi3=vi2, main=var,
                 xticks=xticks,
                 xlabel=xlabels[i],
                 legend=True)
    i += 1

# %%plt
mod = 'mlp'
d = 'sparse'

for var in variables:
    vi = pd.read_csv(f"../results/{mod}_{d}_{var}_via.csv", index_col=False, header=1).iloc[:, 1:]
    plot_via(vi)
# %%
fig, ax = plt.subplots(figsize=(7, 7))
i = 0
n = ['Regularized', 'Parallel Physics', 'Naive']
colors = ['#2c7bb6', '#abd9e9', '#fdae61']
marker = ['.', '^', '*', '+', 'p']

for var in variables:
    vias = np.zeros((3, 2))
    for d in data_use:
        vi = pd.read_csv(f"../resultsN2/reg_{d}_{var}_via.csv", index_col=False, header=1)  # .iloc[:,1:]
        vi2 = pd.read_csv(f"../resultsN2/res2_{d}_{var}_via.csv", index_col=False, header=1)  # .iloc[:,1:]
        vi3 = pd.read_csv(f"../resultsN2/mlp_{d}_{var}_via.csv", index_col=False, header=1)  # .iloc[:,1:]

        vm = np.mean(vi.iloc[:, 1:], axis=1)
        print('Regularized')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[0, 0] = np.std(vm)

        vm = np.mean(vi.iloc[:, 1:], axis=0)
        print('Regularized')
        print("Yeary standard dev in GPP predictions at values of ", var)
        print(np.std(vm))
        vias[0, 1] = np.std(vm)

        vm = np.mean(vi2.iloc[:, 1:], axis=1)
        print('Parallel Physics')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[1, 0] = np.std(vm)

        vm = np.mean(vi2.iloc[:, 1:], axis=1)
        print('Parallel Physics')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[1, 1] = np.std(vm)

        vm = np.mean(vi3.iloc[:, 1:], axis=1)
        print('Naive')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[2, 0] = np.std(vm)

        vm = np.mean(vi3.iloc[:, 1:], axis=1)
        print('Naive')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[2, 1] = np.std(vm)

    ax.scatter(vias[0, 1], vias[0, 0], label=var, color='#fdae61', marker=marker[i], s=300)
    ax.scatter(vias[1, 1], vias[1, 0], label=var, color='#abd9e9', marker=marker[i], s=300)
    ax.scatter(vias[2, 1], vias[2, 0], label=var, color='#2c7bb6', marker=marker[i], s=300)
    #    for i, txt in enumerate(n):
    #        ax.annotate(txt, (vias[:,1][i]+0.1, vias[:,0][i]+0.1), size=12, color="black")

    i += 1

ax.set_xlabel("$\sigma(\overline{\widehat{\mathrm{GPP}}})$", fontsize=18)
ax.set_ylabel("$\overline{\sigma(\widehat{\mathrm{GPP}}_{i})}$", fontsize=18)

# legend
label_column = variables
label_row = n
color = np.array([colors, ] * 5).transpose()
rows = [mpatches.Patch(color=color[i, 0]) for i in range(3)]
columns = [plt.plot([], [], marker[i], markerfacecolor='w',
                    markeredgecolor='k', markersize=12)[0] for i in range(5)]

plt.legend(rows + columns, label_row + label_column, fontsize=12, loc='lower right')

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(18)
    # tick.label.set_fontfamily({'font.serif':'Palatino'})
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(18)

fig.savefig(f'../plots/VIA_summarized.pdf', bbox_inches='tight', dpi=300, format='pdf')

# %%