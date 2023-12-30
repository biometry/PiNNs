#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:47:12 2022

@author: Marieke_Wesselkamp
"""

import sys, os
sys.path.append("/Users/mw1205/PycharmProjects/physics_guided_nn/code")
os.chdir("/Users/mw1205/PycharmProjects/physics_guided_nn/code")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import get_seasonal_data

current_dir = '/Users/mw1205/PycharmProjects/physics_guided_nn'

def plot_performance(performance, data_use, experiment, log=False,
                     colors=['#efe645','#e935a1','#00e3ff','#e1562c','#537eff', '#5954d6']):

    plt.rc('font',**{'family':'sans-serif'})
    
    fig,ax = plt.subplots(figsize=(9,9))

    if log:
        performance = np.log(performance)

    bp = ax.boxplot(performance, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        
    # whiskers
    for whisker in bp['whiskers']:
        whisker.set(color ='black',
                    linewidth = 2.5,
                    linestyle =":")
    for median in bp['medians']:
        median.set(color ='black',
                   linewidth = 2)
    
    if not log:
        if (experiment == 1) & (data_use == 'full'):
            ax.set_ylim(0.5,1.3)
        if (experiment == 1) & (data_use == 'sparse'):
            ax.set_ylim(0.5,1.3)
        if (experiment != 1) & (data_use == 'full'):
            ax.set_ylim(0.5,4.5)
        if (experiment != 1) & (data_use == 'sparse'):
            ax.set_ylim(0.5,4.5)
    
    if log:
        ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=22)
    else:
        ax.set_ylabel('Mean absolute error [g C m$^{-2}$ day$^{-1}$]', fontsize=26)
        
    ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrised', 'Domain\nAdaptation'],
                       rotation = 45)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(24)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(24)
    plt.tight_layout()
    plt.show()
    
    if log:
        fig.savefig(os.path.join(current_dir, f"plots/accuracy/exp{experiment}performance_{data_use}_log.pdf"),
                    bbox_inches = 'tight', dpi=200, format='pdf')
    else:
        fig.savefig(os.path.join(current_dir, f"plots/accuracy/exp{experiment}performance_{data_use}.pdf"),
                    bbox_inches = 'tight', dpi=200, format='pdf')

    plt.close()

def observed_seasonal_means(data_use = 'full', model = 'mlp', prediction_scenario='exp2'):

    days, days_yp, var_ranges, mn, std  = get_seasonal_data(data_use=data_use, model=model, prediction_scenario=prediction_scenario)

    gpp_means = []
    for mon, df in days_yp.items():
        #day_yp = days_yp['mar']
        GPP_mean = [col for col in df.columns if col.startswith('GPP')]
        gpp_means.append(df[GPP_mean].mean().values.mean())

    if model == 'mlp':
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']
    elif model == 'res':
        variables = ['GPPp', 'ETp', 'SWp']

    var_means = []
    for v in variables:
        var_mean = []
        for mon, df in days.items():
            #day_yp = days_yp['mar']
            vars = [col for col in df.columns if col.startswith(v)]
            var_mean.append(df[vars].mean().values.mean())
        var_means.append(var_mean)
    var_means.append(gpp_means)
    all_means = pd.DataFrame(np.transpose(np.array(var_means)), columns = variables+['GPP'])

    return all_means, var_ranges, mn, std

def get_predictions(exp, mod):

    if exp == 1:
        preds_mlp = pd.read_csv(f"../results_exp{exp}/mlp_full_eval_preds_test.csv",
                                index_col=False)
        preds_mlp_sparse = pd.read_csv(f"../results_exp{exp}/mlp_sparse_eval_preds_test.csv",
                                       index_col=False)
        preds_pgnn = pd.read_csv(f"../results_exp{exp}/{mod}_eval_preds_full_test.csv",
                                 index_col=False)
        preds_pgnn_sparse = pd.read_csv(f"../results_exp{exp}/{mod}_eval_preds_sparse_test.csv",
                                        index_col=False)

        hyy = pd.read_csv(f"../data/hyytialaF_full.csv",
                          index_col=False)
        hyy.index = pd.DatetimeIndex(hyy['date'])

        hyy_sparse = pd.read_csv(f"../data/hyytialaF_sparse.csv",
                                 index_col=False)
        hyy_sparse.index = pd.DatetimeIndex(hyy_sparse['date'])

        if mod != 'mlpDA1':
            hyy = hyy[hyy.index.year == 2008][1:]
            hyy_sparse = hyy_sparse[hyy_sparse.index.year == 2008][1:]
        else:
            hyy = hyy[hyy.index.year == 2008][1:]
            hyy_sparse = hyy_sparse[hyy_sparse.index.year == 2008][1:]
            preds_pgnn = preds_pgnn[1:]
            preds_pgnn_sparse = preds_pgnn_sparse[1:]

        obs = hyy[['GPP']]
        preds_preles = hyy[['GPPp']]
        preds_preles_sparse = hyy_sparse[['GPPp']]

    else:
        preds_mlp = pd.read_csv(f"../results_exp{exp}/{exp}mlp_full_eval_preds_test.csv",
                                index_col=False)
        preds_mlp_sparse = pd.read_csv(f"../results_exp{exp}/{exp}mlp_sparse_eval_preds_test.csv",
                                       index_col=False)
        preds_pgnn = pd.read_csv(f"../results_exp{exp}/{exp}{mod}_eval_preds_full_test.csv",
                                 index_col=False)
        preds_pgnn_sparse = pd.read_csv(f"../results_exp{exp}/{exp}{mod}_eval_preds_sparse_test.csv",
                                        index_col=False)

        hyy = pd.read_csv(f"../data/allsitesF_exp{exp}_full.csv",
                          index_col=False)
        hyy.index = pd.DatetimeIndex(hyy['date'])

        hyy_sparse = pd.read_csv(f"../data/allsitesF_exp{exp}_sparse.csv",
                                 index_col=False)
        hyy_sparse.index = pd.DatetimeIndex(hyy_sparse['date'])

        hyy = hyy[((hyy.index.year == 2005) | (hyy.index.year == 2008)) & (hyy.site == "h").values]
        preds_mlp = preds_mlp[(hyy.index.year == 2008)]
        preds_mlp_sparse = preds_mlp_sparse[(hyy.index.year == 2008)]
        preds_pgnn = preds_pgnn[(hyy.index.year == 2008)]
        preds_pgnn_sparse = preds_pgnn_sparse[(hyy.index.year == 2008)]

        hyy = hyy[(hyy.index.year == 2008)]
        hyy_sparse = hyy_sparse[(hyy_sparse.index.year == 2008) & (hyy_sparse.site == "h").values]

        obs = hyy[['GPP']]
        preds_preles = hyy[['GPPp']]
        preds_preles_sparse = hyy_sparse[['GPPp']]

    preds_mlp = preds_mlp.drop(columns=['Unnamed: 0'])
    preds_mlp_sparse = preds_mlp_sparse.drop(columns=['Unnamed: 0'])
    preds_pgnn = preds_pgnn.drop(columns=['Unnamed: 0'])
    preds_pgnn_sparse = preds_pgnn_sparse.drop(columns=['Unnamed: 0'])

    return preds_mlp, preds_mlp_sparse, preds_pgnn, preds_pgnn_sparse, obs, preds_preles, preds_preles_sparse

def plot_via(d = "full", prediction_scenario = 'exp2', current_dir ='', save=True):

    months = ["dec", "mar", "jun", "sep"]
    days = ["Spring", "Summer", "Autum", "Winter"]
    colors = ["lightblue","darkgreen",  "lightgreen", "darkblue"]
    var = ["$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]",
           "$D$ [kPa]", "$D$ [kPa]", "$D$ [kPa]", "$D$ [kPa]","$D$ [kPa]",
           "$R$ [mm]", "$R$ [mm]", "$R$ [mm]", "$R$ [mm]","$R$ [mm]",
           "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]","$\phi$ [mol m$^{-2}$ d$^{-1}$]",
           "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$"]

    cols = ["PRELES", "Naive", "Parallel\n Physics", "Regularized", "Domain\n Adaptation"]
    all_means, var_ranges, mn, std = observed_seasonal_means(data_use=d, prediction_scenario=prediction_scenario)

    fig = plt.figure(figsize=(80,12))
    widths = [i for i in np.repeat(3, 5)]
    heights = [i for i in np.repeat(3, 5)]

    gs = fig.add_gridspec(5, 5, width_ratios = widths, height_ratios=heights, wspace=0.5, hspace=0.8)
    (ax1, ax2, ax3, ax4, ax5) , \
    (ax6, ax7, ax8, ax9, ax10) , \
    (ax11, ax12, ax13, ax14, ax15) , \
    (ax16 , ax17, ax18, ax19, ax20),\
        (ax21 , ax22, ax23, ax24, ax25) = gs.subplots() #sharey='row'

    def plot_variable(v,ax, mod, d, prediction_scenario):
        for i in range(4):
            vi3 = np.array(pd.read_csv(os.path.join(current_dir,f"results_{prediction_scenario}/via/{mod}_{d}_{v}_via_cond_{months[i]}.csv"), index_col=False).iloc[:,1:])
            vi3_m = vi3.mean(axis=1)
            vi3_q = np.quantile(vi3, (0.05, 0.95), axis=1)
            ax.fill_between(var_ranges[v], vi3_q[0],vi3_q[1],color=colors[i], alpha=0.2)
            ax.plot(var_ranges[v], vi3_m, color=colors[i], label=days[i])
            ax.plot(all_means[v][i], all_means['GPP'][i], marker='x', markersize=10, color=colors[i])

    j=0
    axs = (ax1, ax6, ax11, ax16, ax21)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'preles', d, prediction_scenario)
        j += 1

    j=0
    axs = (ax2, ax7, ax12, ax17, ax22)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'mlp', d, prediction_scenario)
        j += 1

    j=0
    axs = (ax3, ax8, ax13, ax18, ax23)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res2', d, prediction_scenario)
        j += 1

    j=0
    axs = (ax4, ax9, ax14, ax19, ax24)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'reg', d, prediction_scenario)
        j += 1

    j=0
    axs = (ax5, ax10, ax15, ax20, ax25)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'mlpDA', d, prediction_scenario)
        j += 1

    i = 0
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlabel(f"{var[i]}", size=18)
        #ax.set_ylabel(f"{ylabels[i]}", size=20)
        i += 1
        #ax.label_outer()

    axs = fig.get_axes()
    for i in range(len(cols)):
        axs[i].set_title(cols[i], size=24)

    fig.text(0.02, 0.5, 'Conditional GPP Predictions [g C m$^{-2}$ day$^{-1}$]', va='center', rotation='vertical', size=20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = (0.01, 0.0),ncol=4, fontsize=20) # loc=(0., 0.05)

    if save:
        fig.savefig(os.path.join(current_dir, f'plots/via/via_{prediction_scenario}_{d}.pdf'),  dpi=300, format='pdf')
        fig.savefig(os.path.join(current_dir, f'plots/via/via_{prediction_scenario}_{d}.jpg'),  dpi=300, format='jpg')

    fig.show()
    plt.close()

def plot_via_biascorrection(current_dir =''):

    months = ["dec", "mar", "jun", "sep"]
    days = ["Spring", "Summer", "Autum", "Winter"]
    colors = ["lightblue","darkgreen",  "lightgreen", "darkblue"]
    var = ["$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]","$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]",
           "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]",
           "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]"]

    cols = ["Temporal\n full", "Temporal\n sparse", "Spatial\n full", "Spatial\n sparse", "Spatial-\ntemporal\n full", "Spatial-\ntemporal\n sparse"]

    fig = plt.figure(figsize=(80,12))
    widths = [i for i in np.repeat(4, len(cols))]
    heights = [i for i in np.repeat(3, 3)]

    gs = fig.add_gridspec(3, len(cols), width_ratios = widths, height_ratios=heights, wspace=0.5, hspace=0.8)
    (ax1, ax2, ax3, ax4, ax5, ax6) , \
    (ax7, ax8, ax9, ax10, ax11, ax12) , \
    (ax13, ax14, ax15, ax16, ax17, ax18)  = gs.subplots()

    all_means, var_ranges, mn, std = observed_seasonal_means(model='res')
    def plot_variable(v,ax, mod, d, prediction_scenario):
        for i in range(4):
            all_means, var_ranges, mn, std = observed_seasonal_means(data_use=d,model='res',
                                                                     prediction_scenario=prediction_scenario)

            vi3 = np.array(pd.read_csv(os.path.join(current_dir,f"results_{prediction_scenario}/via/{mod}_{d}_{v}_via_cond_{months[i]}.csv"), index_col=False).iloc[:,1:])
            vi3_m = vi3.mean(axis=1)
            vi3_q = np.quantile(vi3, (0.05, 0.95), axis=1)
            ax.fill_between(var_ranges[v], vi3_q[0],vi3_q[1],color=colors[i], alpha=0.2)
            ax.plot(var_ranges[v], vi3_m, color=colors[i], label=days[i])
            ax.plot(all_means[v][i], all_means['GPP'][i], marker='x', markersize=10, color=colors[i])

    j=0
    axs = (ax1, ax7, ax13)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'full', 'exp1')
        j += 1

    j=0
    axs = (ax2, ax8, ax14)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'sparse', 'exp1')
        j += 1

    j=0
    axs = (ax3, ax9, ax15)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'full', 'exp2')
        j += 1

    j=0
    axs = (ax4, ax10, ax16)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'sparse', 'exp2')
        j += 1

    j = 0
    axs = (ax5, ax11, ax17)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j], 'res', 'sparse', 'exp3')
        j += 1

    j = 0
    axs = (ax6, ax12, ax18)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j], 'res', 'sparse', 'exp3')
        j += 1


    i = 0
    for ax in fig.get_axes():
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlabel(f"{var[i]}", size=18)
        #ax.set_ylabel(f"{ylabels[i]}", size=20)
        i += 1
        #ax.label_outer()

    axs = fig.get_axes()
    for i in range(len(cols)):
        axs[i].set_title(cols[i], size=24)

    fig.text(0.02, 0.5, 'Conditional Predictions', va='center', rotation='vertical', size=20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = (0.01, 0.0),ncol=4, fontsize=20) # loc=(0., 0.05)

    fig.savefig(os.path.join(current_dir, f'plots/via/via_res.pdf'),  dpi=300, format='pdf')
    fig.savefig(os.path.join(current_dir, f'plots/via/via_res.jpg'),  dpi=300, format='jpg')
    fig.show()
    plt.close()

def plot_predictions(obs, y_preles, y_mlp, y_pgnn, model = 'PGNN', save_to=''):

    x = np.arange(len(obs))

    y_mlp_m = y_mlp.mean(axis=1)
    y_pgnn_m = y_pgnn.mean(axis=1)
    y_mlp_q = [np.min(y_mlp, axis=1), np.max(y_mlp, axis=1)]
    y_pgnn_q = [np.min(y_pgnn, axis=1), np.max(y_pgnn, axis=1)]


    error_pgnn = np.subtract(obs.to_numpy().squeeze(), y_pgnn_m.to_numpy())
    error_mlp = np.subtract(obs.to_numpy().squeeze(), y_mlp_m.to_numpy())

    plt.rc('font', **{'family': 'sans-serif'})
    plt.rcParams.update({'font.size': 22})
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(9, 8))

    axs[0].plot(x, obs, color='black', label='Observed', linewidth=0.9, alpha=0.8)
    axs[0].plot(x, y_preles, color='gray', label='Preles', linewidth=0.9, alpha=0.8)
    axs[0].fill_between(x, y_pgnn_q[0], y_pgnn_q[1], color='blue', alpha=0.7)
    axs[0].plot(x, y_pgnn_m, color='blue', label=model, alpha = 0.8, linewidth=0.9)
    axs[0].fill_between(x, y_mlp_q[0], y_mlp_q[1], color='salmon', alpha=0.7)
    axs[0].plot(x, y_mlp_m, color='salmon', label='MLP', alpha = 0.8, linewidth=0.9)
    axs[0].legend(loc='upper left', fontsize=20)
    axs[0].set_ylabel('GPP [g C m-2 s-1]')

    axs[1].plot(x, error_pgnn, color="blue", linewidth=1.0)
    axs[1].plot(x, error_mlp, color="salmon", linewidth=1.0)
    axs[1].axhline(y=0, color='black', linestyle="--", linewidth=0.9)
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Time [days]')

    plt.tight_layout()
    fig.savefig(os.path.join('../plots/temporalprediction', f'{save_to}.pdf'), bbox_inches = 'tight', dpi=200, format='pdf')
    fig.savefig(os.path.join('../plots/temporalprediction',f'{save_to}.jpg'), bbox_inches = 'tight', dpi=200, format='jpg')
    plt.close()

def plot_prediction_correlations(y_pgnn_full, y_pgnn_sparse, obs, save_to=''):


    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 22})

    from scipy import stats

    obs = obs.squeeze()
    y_pgnn_full = y_pgnn_full.mean(axis=1).squeeze()
    y_pgnn_sparse = y_pgnn_sparse.mean(axis=1).squeeze()

    output = stats.binned_statistic(obs, y_pgnn_full, statistic='mean', bins=10)
    y_pgnn_full_mean = output.statistic
    output = stats.binned_statistic(obs, y_pgnn_full, statistic='std', bins=10)
    y_pgnn_full_std = output.statistic

    output = stats.binned_statistic(obs, y_pgnn_sparse, statistic='mean', bins=10)
    y_pgnn_sparse_mean = output.statistic
    output = stats.binned_statistic(obs, y_pgnn_sparse, statistic='std', bins=10)
    y_pgnn_sparse_std = output.statistic
    bins = output.bin_edges


    x = np.linspace(np.nanmin(np.concatenate((y_pgnn_full_mean, y_pgnn_sparse_mean, bins[:-1]))),
                    np.nanmax(np.concatenate((y_pgnn_full_mean, y_pgnn_sparse_mean, bins[:-1]))), 100)
    # Calculate the corresponding y-values (y = x)
    y = x
    # Create the scatterplot
    plt.plot(x, y, linestyle = '--', color = 'black')
    plt.errorbar(bins[:-1], y_pgnn_full_mean, yerr=y_pgnn_full_std, linestyle='', marker='o',
                     markersize = 15, alpha=0.8, color='blue', label='Full')
    plt.errorbar(bins[:-1], y_pgnn_sparse_mean, yerr=y_pgnn_sparse_std,  linestyle='', marker='o',
                     markersize = 15, alpha=0.7, color='salmon', label='Sparse')
    plt.ylabel("Predicted GPP [g C m-2 s-1]")
    plt.xlabel("Observed GPP [g C m-2 s-1]")
    #plt.ylim((0,np.nanmax(np.concatenate((y_pgnn_full, y_pgnn_sparse, observation[0]))))) #np.nanmin(np.concatenate((y_pgnn_full, y_pgnn_sparse, observation)))
    #plt.xlim((0,np.nanmax(np.concatenate((y_pgnn_full, y_pgnn_sparse, observation[0])))))

    plt.legend(loc='upper left', fontsize=20) #loc='upper left', bbox_to_anchor=(1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join('../plots/correlation', f'{save_to}.pdf'), bbox_inches = 'tight', dpi=200, format='pdf')
    plt.savefig(os.path.join('../plots/correlation', f'{save_to}.jpg'), bbox_inches='tight', dpi=200, format='jpg')
    plt.close()

def plot_prediction_correlations2(y_pgnn, obs, colors =['#efe645','#e935a1','#00e3ff','#e1562c','#537eff', '#5954d6'], save_to=''):


    plt.figure(figsize=(9, 9))
    plt.rcParams.update({'font.size': 24})

    from scipy import stats

    obs = obs.squeeze()

    means = []
    stds = []

    for pgnn in y_pgnn:

        pgnn = pgnn.mean(axis=1).squeeze()

        output = stats.binned_statistic(obs, pgnn, statistic='mean', bins=10)
        means.append(output.statistic)
        output = stats.binned_statistic(obs, pgnn, statistic='std', bins=10)
        stds.append(output.statistic)
        bins = output.bin_edges


    x = np.linspace(np.nanmin(np.concatenate((np.concatenate(means), bins[:-1]))),
                    np.nanmax(np.concatenate((np.concatenate(stds), bins[:-1]))), 100)
    # Calculate the corresponding y-values (y = x)
    y = x
    # Create the scatterplot
    markers = ['o', 's', 'v', 'D', 'P', 'X']
    labels = ['Preles', 'MLP', 'Bias\n Correction', 'Parallel\nPhysics', 'Regularized\nPhysics', 'Domain\nAdaptation']

    plt.plot(x, y, linestyle = '--', color = 'black')
    for i in range(len(means)):
        plt.errorbar(bins[:-1], means[i], yerr=stds[i], linestyle='-', marker=markers[i],xuplims=True, xlolims=True,
                         markersize = 13, alpha=0.89, color=colors[i], label=labels[i])

    plt.ylabel("Predicted GPP [g C m-2 s-1]")
    plt.xlabel("Observed GPP [g C m-2 s-1]")

    plt.legend(loc='upper left', fontsize=20) #loc='upper left', bbox_to_anchor=(1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join('../plots/correlation', f'{save_to}.pdf'), bbox_inches = 'tight', dpi=200, format='pdf')
    plt.savefig(os.path.join('../plots/correlation', f'{save_to}.jpg'), bbox_inches='tight', dpi=200, format='jpg')
    plt.close()

def load_performances(exp, mod, data_use):

    if exp == 1:
        performance = pd.read_csv(f"../results_exp{exp}/{mod}_eval_{data_use}_performance.csv",
                                  index_col=False)
    else:
        performance = pd.read_csv(f"../results_exp{exp}/{exp}{mod}_eval_{data_use}_performance.csv",
                                  index_col=False)

    return performance

if __name__ == '__main__':

    mods = ['preles', 'mlp', 'res', 'res2', 'reg', 'mlpDA1']  # , "emb"
    data_use = ['full', 'sparse']
    experiments = [1,2,3]
    perf = np.zeros((4, len(mods)))
    performances_all = []

    for exp in experiments:
        for d in data_use:
            i = 0
            for mod in mods:

                performance = load_performances(exp, mod, d)

                if mod != 'preles':
                    perf[:, i] = performance.iloc[:, 4]
                else:
                    perf[:, i] = performance.iloc[:, 2]

                i += 1
            performances_all.append(perf.copy())

            plot_performance(perf, data_use=d, experiment=exp, log=False)

            print(f'Median prediction performance {d} experiement {exp}:',
                  list(zip(np.round(np.median(perf, axis=0), 2), np.round(np.std(perf, axis=0), 2))))


    plot_via(d="full", prediction_scenario='exp1', current_dir=current_dir, save=True)
    plot_via(d="sparse", prediction_scenario='exp1', current_dir=current_dir, save=True)
    plot_via(d="full", prediction_scenario='exp2', current_dir=current_dir, save=True)
    plot_via(d="sparse", prediction_scenario='exp2', current_dir=current_dir, save=True)
    plot_via(d="full", prediction_scenario='exp3', current_dir=current_dir, save=True)
    plot_via(d="sparse", prediction_scenario='exp3', current_dir=current_dir, save=True)

    plot_via_biascorrection(current_dir=current_dir)

    #==================#
    # Plot predictions #
    #==================#

    exps = [1,2,3]
    mods = ['res2', 'reg', 'mlpDA1']
    mods_names = ['Parallel\nPhysics', 'Regularized\nPhysics', 'Domain\nAdaptation']

    for exp in exps:

        print('Plot predictions: Experiment ', exp)

        for mod in mods:

            print('Plot predictions: Model ', mod)

            preds_mlp, preds_mlp_sparse, preds_pgnn, preds_pgnn_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(exp, mod)

            # Plot predictions:
            plot_predictions(obs, preds_preles, preds_mlp, preds_pgnn, model=mods_names[(exp - 1)],
                             save_to=f'predictions_exp{exp}_{mod}')


            plot_prediction_correlations(preds_pgnn, preds_pgnn_sparse, obs,
                                         save_to=f'predictions_correlations_exp{exp}_{mod}')

            plot_prediction_correlations(preds_preles, preds_preles_sparse, obs,
                                         save_to=f'predictions_correlations_exp{exp}_preles')

            plot_prediction_correlations(preds_mlp, preds_mlp_sparse, obs,
                                         save_to=f'predictions_correlations_exp{exp}_mlp')

    #==================#
    for exp in exps:

        preds_mlp, preds_mlp_sparse, preds_res, preds_res_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp = exp, mod = 'res')
        preds_mlp, preds_mlp_sparse, preds_res2, preds_res2_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp = exp, mod = 'res2')
        preds_mlp, preds_mlp_sparse, preds_reg, preds_reg_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp=exp, mod='reg')
        preds_mlp, preds_mlp_sparse, preds_mlpDA1, preds_mlpDA1_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp=exp, mod='mlpDA1')

        assembled_preds_full = [preds_preles, preds_mlp, preds_res,preds_res2, preds_reg, preds_mlpDA1]
        assembled_preds_sparse = [preds_preles_sparse, preds_mlp_sparse, preds_res_sparse,preds_res2_sparse, preds_reg_sparse, preds_mlpDA1_sparse]

        plot_prediction_correlations2(assembled_preds_full, obs,
                                        save_to=f'predictions_correlations_exp{exp}_all_full')
        plot_prediction_correlations2(assembled_preds_sparse, obs,
                                        save_to=f'predictions_correlations_exp{exp}_all_sparse')
