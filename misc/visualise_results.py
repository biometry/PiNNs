#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:47:12 2022

@author: Marieke_Wesselkamp
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_seasonal_data


def load_performances(exp, mod, data_use):

    if exp == 1:
        performance = pd.read_csv(f"../temporal/analysis/results/{mod}_eval_{data_use}_performance.csv",
                                  index_col=False)
    elif exp == 2:
        performance = pd.read_csv(f"../spatial/analysis/results/{exp}{mod}_eval_{data_use}_performance.csv",
                                  index_col=False)
    else:
         performance = pd.read_csv(f"../spatio_temporal/analysis/results/{exp}{mod}_eval_{data_use}_performance.csv",
                                  index_col=False)

    return performance

def plot_performance(performance, data_use, experiment, log=False,
                     colors=['#efe645','#e935a1','#00e3ff','#e1562c','#537eff', '#5954d6', 'gray']):

    plt.rc('font',**{'family':'sans-serif'})
    
    fig,ax = plt.subplots(figsize=(10,10))

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
        if (experiment == 1) :
            ax.set_ylim(0.5,1.3)
        elif (experiment == 2):
            ax.set_ylim(0.5,2.2)
        elif (experiment == 3):
            ax.set_ylim(0.5,3.1)
    
    if log:
        ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=28)
    else:
        ax.set_ylabel('Mean absolute error [g C m$^{-2}$ day$^{-1}$]', fontsize=28)
        
    ax.set_xticklabels(['$\mathbf{PRELES}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrised', 'Domain\nAdaptation', 'Embedding'],
                       rotation = 45)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(26)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(26)
    plt.tight_layout()
    plt.show()
    
    if log:
        fig.savefig(os.path.join(current_dir, f"plots/accuracy/exp{experiment}performance_{data_use}_log.pdf"),
                    bbox_inches = 'tight', dpi=200, format='pdf')
    else:
        fig.savefig(os.path.join(current_dir, f"plots/accuracy/exp{experiment}performance_{data_use}.pdf"),
                    bbox_inches = 'tight', dpi=200, format='pdf')

    plt.close()

def performance_plots(exps=[1, 2, 3],
                          mods=['preles', 'mlp', 'res', 'res2', 'reg', 'mlpDA1', 'emb'],
                          data_use=['full', 'sparse']):

    perf = np.zeros((4, len(mods)))
    performances_all = []

    for exp in exps:
        print('Plot performances: Experiment ', exp)

        for d in data_use:

            print('Plot performances: Data use ', d)
            i = 0
            for mod in mods:
                print('Plot performances: Model ', mod)

                performance = load_performances(exp, mod, d)

                if mod != 'preles':
                    perf[:, i] = performance.iloc[:, 4]
                else:
                    perf[:, i] = performance.iloc[:, 2]

                i += 1

            performances_all.append(perf.copy())
            plot_performance(perf, data_use=d, experiment=exp, log=False)
            median_performance = np.median(perf, axis=0)
            best_performance = np.where(median_performance == np.min(median_performance))[0]

            #print(f'Best model in Experiment {exp} and scenario {d}:', mods[best_performance])


def observed_seasonal_means(data_use = 'full', model = 'mlp', prediction_scenario='exp2', rescale=False):


    days, days_yp, var_ranges, mn, std  = get_seasonal_data(data_use=data_use, model=model, prediction_scenario=prediction_scenario)

    if model == 'res':
        variables = ['GPPp', 'ETp', 'SWp']
        mn = dict(zip(['GPPp', 'ETp', 'SWp'], mn[:3].to_numpy()))
        std = dict(zip(['GPPp', 'ETp', 'SWp'], std[:3].to_numpy()))
    else:
        variables = ['PAR', 'Tair', 'VPD', 'Precip', 'fapar']

    if rescale:
        for k, v in days_yp.items():
            for var in variables:
                days_yp[k][var] = (v[var] * std[var]) + mn[var]

    gpp_means = []
    for mon, df in days_yp.items():
        #day_yp = days_yp['mar']
        GPP_mean = [col for col in df.columns if col.startswith('GPP')]
        gpp_means.append(df[GPP_mean].mean().values.mean())

    var_means = []
    for v in variables:
        var_mean = []
        for mon, df in days.items():
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



        if (mod != 'mlpDA1'):
            hyy = hyy[((hyy.index.year == 2008)) & (hyy.site == "h").values]
        else:
            hyy = hyy[((hyy.index.year == 2008) | (hyy.index.year==2005)) & (hyy.site == "h").values]
            preds_pgnn = preds_pgnn[(hyy.index.year == 2008)]
            hyy = hyy[(hyy.index.year == 2008)]
        #preds_mlp = preds_mlp[(hyy.index.year == 2008)]
        #preds_mlp_sparse = preds_mlp_sparse[(hyy.index.year == 2008)]
        #preds_pgnn = preds_pgnn[(hyy.index.year == 2008)]
        #preds_pgnn_sparse = preds_pgnn_sparse[(hyy.index.year == 2008)]

        hyy_sparse = hyy_sparse[(hyy_sparse.index.year == 2008) & (hyy_sparse.site == "h").values]

        obs = hyy[['GPP']]
        preds_preles = hyy[['GPPp']]
        preds_preles_sparse = hyy_sparse[['GPPp']]

    preds_mlp = preds_mlp.drop(columns=['Unnamed: 0'])
    preds_mlp_sparse = preds_mlp_sparse.drop(columns=['Unnamed: 0'])
    preds_pgnn = preds_pgnn.drop(columns=['Unnamed: 0'])
    preds_pgnn_sparse = preds_pgnn_sparse.drop(columns=['Unnamed: 0'])

    return preds_mlp, preds_mlp_sparse, preds_pgnn, preds_pgnn_sparse, obs, preds_preles, preds_preles_sparse

def plot_via(d = "full", prediction_scenario = 'exp2',rescale = False, current_dir ='', save=True):

    months = ["mar", "jun", "sep", "dec"]
    days = ["Spring", "Summer", "Autum", "Winter"]
    colors = ["lightblue","darkgreen",  "lightgreen", "darkblue"]
    var = ["$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]",
           "$D$ [kPa]", "$D$ [kPa]", "$D$ [kPa]", "$D$ [kPa]","$D$ [kPa]",
           "$R$ [mm]", "$R$ [mm]", "$R$ [mm]", "$R$ [mm]","$R$ [mm]",
           "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]","$\phi$ [mol m$^{-2}$ d$^{-1}$]",
           "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$"]

    cols = ["PRELES", "Naive", "Parallel\n Physics", "Regularized", "Domain\n Adaptation"]
    all_means, var_ranges, mn, std = observed_seasonal_means(data_use=d, prediction_scenario=prediction_scenario)

    if rescale:
        var_ranges['Tair'] = var_ranges['Tair'] * std['Tair'] + mn['Tair']
        var_ranges['VPD'] = var_ranges['VPD'] * std['VPD'] + mn['VPD']
        var_ranges['Precip'] = var_ranges['Precip'] * std['Precip'] + mn['Precip']
        var_ranges['fapar'] = var_ranges['fapar'] * std['fapar'] + mn['fapar']
        var_ranges['PAR'] = var_ranges['PAR'] * std['PAR'] + mn['PAR']

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

def plot_via_seasonal_single(d = "full", prediction_scenario = 'exp1',current_dir =''):

    cols = ["PRELES", "Naive", "Parallel\n Physics", "Regularized", "Domain\n Adaptation"]

    all_means, var_ranges, mn, std = observed_seasonal_means(data_use=d, prediction_scenario=prediction_scenario)
    all_means.index = ['mar', 'jun', 'sep', 'dec']
    def plot_variable(variable ,data_use, month, prediction_scenario):

        plt.rc('font', **{'family': 'sans-serif'})
        plt.rcParams.update({'font.size': 30})
        var = {'Tair':"$T$ [$\degree$C]", 'VPD':"$D$ [kPa]", "Precip":"$R$ [mm]", 'PAR':"$\phi$ [mol m$^{-2}$ d$^{-1}$]",
               'fapar':"$f_{aPPFD}$"}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))
        models = ['preles', 'mlp', 'res2', 'reg', 'mlpDA']
        colors = ['#efe645','#e935a1','#e1562c','#537eff', '#5954d6']
        for i in range(len(models)):
            vi3 = np.array(pd.read_csv(os.path.join(current_dir,f"results_{prediction_scenario}/via/{models[i]}_{data_use}_{variable}_via_cond_{month}.csv"), index_col=False).iloc[:,1:])
            vi3_m = vi3.mean(axis=1)
            vi3_q = np.quantile(vi3, (0.05, 0.95), axis=1)
            ax.fill_between(var_ranges[variable], vi3_q[0],vi3_q[1],color=colors[i], alpha=0.35)
            ax.plot(var_ranges[variable], vi3_m, color=colors[i], label=models[i], linewidth=1.5)
            ax.plot(all_means[variable].loc[month], all_means['GPP'].loc[month], marker='x', markersize=15, color='black')
        i = 0
        for ax in fig.get_axes():
            ax.tick_params(axis='x', labelsize=30)
            ax.tick_params(axis='y', labelsize=30)
            ax.set_xlabel(f"{var[variable]}", size=30,labelpad=25)
            # ax.set_ylabel(f"{ylabels[i]}", size=20)
            i += 1
            # ax.label_outer()
        ax.set_ylabel('GPP Predictions [g C m$^{-2}$ day$^{-1}$]', labelpad=25, va='center', rotation='vertical', size=26)
        fig.tight_layout()
        fig.savefig(os.path.join(current_dir, f'plots/via/seasonal_via_{prediction_scenario}_{d}_{variable}_{month}.pdf'), dpi=300,
                    format='pdf')
        fig.savefig(os.path.join(current_dir, f'plots/via/seasonal_via_{prediction_scenario}_{d}_{variable}_{month}.svg'),
            format='svg')
        fig.savefig(os.path.join(current_dir, f'plots/via/seasonal_via_{prediction_scenario}_{d}_{variable}_{month}.jpg'), dpi=300,
                    format='jpg')

        fig.show()
        plt.close()

    for key, value in var_ranges.items():
        plot_variable(key, d, 'sep', prediction_scenario)
    for key, value in var_ranges.items():
        plot_variable(key, d, 'dec', prediction_scenario)
    for key, value in var_ranges.items():
        plot_variable(key, d, 'mar', prediction_scenario)
    for key, value in var_ranges.items():
        plot_variable(key, d, 'jun', prediction_scenario)

def plot_via_seasonal_together(month, d="full", prediction_scenario='exp1',
                      models = [ 'mlp', 'res2', 'reg', 'mlpDA', 'preles'],
                      colors = ['#e935a1','#e1562c','#537eff', '#5954d6', '#efe645'],
                        plots = "all", current_dir=''):

    all_means, var_ranges, mn, std = observed_seasonal_means(data_use=d, prediction_scenario=prediction_scenario)
    all_means.index = ['mar', 'jun', 'sep', 'dec']

    def plot_variable(ax, variable, data_use, month, prediction_scenario, models, colors):
        if models[0]== 'embtest':
            labs = ['Embedding$_{raw}$', 'Embedding$_{preles}$', 'PRELES']
        else:
            labs = ["Naive", "Parallel\n Physics", "Regularised\n Physics", "Domain\n Adaptation", "PRELES"]
        for i in range(len(models)):
            vi3 = np.array(pd.read_csv(
                os.path.join(current_dir, f"results_{prediction_scenario}/via/{models[i]}_{data_use}_{variable}_via_cond_{month}.csv"),
                index_col=False).iloc[:, 1:])
            vi3_m = vi3.mean(axis=1)
            vi3_q = np.quantile(vi3, (0.05, 0.95), axis=1)
            ax.fill_between(var_ranges[variable], vi3_q[0], vi3_q[1], color=colors[i], alpha=0.35)
            ax.plot(var_ranges[variable], vi3_m, color=colors[i], label=labs[i], linewidth=1.75)

        #ax.plot(all_means[variable].loc[month], all_means['GPP'].loc[month], marker='x', markersize=15, color='black')

    variables = list(var_ranges.keys())
    var = {'Tair': "$T$ [$\degree$C]", 'VPD': "$D$ [kPa]", "Precip": "$R$ [mm]",
               'PAR': "$\phi$ [mol m$^{-2}$ d$^{-1}$]", 'fapar': "$f_{aPPFD}$"}

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 10), sharey=False,
                                 gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1, 1]})
    figlegend = plt.figure(figsize=(4,4))
    for i in range(2):
        for j in range(3):
            axs = axes[i, j]
            if ((i == 0) and (j == 0)):
                axs.axis('off')
            elif variables:
                variable = variables.pop(0)
                plot_variable(axs, variable, d, month, prediction_scenario, models, colors)
                axs.tick_params(axis='x', labelsize=30)
                axs.tick_params(axis='y', labelsize=30)
                axs.set_xlabel(f"{var[variable]}", size=30, labelpad=20)
                axs.set_ylabel('GPP [g C m$^{-2}$ day$^{-1}$]', labelpad=20, va='center',
                                  rotation='vertical', size=26)
        handles, labels = axs.get_legend_handles_labels()
        figlegend.legend(handles, labels , loc='center')
            # Create a common legend for all subplots
            #handles, labels = ax.get_legend_handles_labels()
            #fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.95, 0.9), fontsize=20)

    fig.tight_layout()
    plt.subplots_adjust(hspace=0.45, wspace=0.5)
    fig.savefig(os.path.join(current_dir, f'plots/via/seasonal_via_{plots}_{prediction_scenario}_{d}_{month}.pdf'), dpi=300,
                        format='pdf')
    figlegend.savefig(os.path.join(current_dir, f'plots/via/via_legend.pdf'), dpi=300, format='pdf')
    fig.savefig(os.path.join(current_dir, f'plots/via/seasonal_via_{plots}_{prediction_scenario}_{d}_{month}.svg'), format='svg')
    fig.savefig(os.path.join(current_dir, f'plots/via/seasonal_via_{plots}_{prediction_scenario}_{d}_{month}.jpg'), dpi=300,
                        format='jpg')
    plt.close()

def plot_via_biascorrection(current_dir =''):

    months = ["mar", "jun", "sep", "dec"]
    days = ["Spring", "Summer", "Autum", "Winter"]
    colors = ["lightblue","darkgreen",  "lightgreen", "darkblue"]
    var = ["$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]","$GPP$\n [g C m$^{-2}$ day$^{-1}$]", "$GPP$\n [g C m$^{-2}$ day$^{-1}$]",
           "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]",
           "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]"]

    cols = ["Temporal\n full", "Temporal\n sparse", "Spatial\n full", "Spatial\n sparse", "Spatio-\ntemporal\n full", "Spatio-\ntemporal\n sparse"]

    fig = plt.figure(figsize=(28,12))
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

    fig.text(0.02, 0.5, 'Predicted GPP [g C m$^{-2}$ day$^{-1}$]', va='center', rotation='vertical', size=20)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc = (0.01, 0.0),ncol=4, fontsize=20) # loc=(0., 0.05)

    fig.savefig(os.path.join(current_dir, f'plots/via/via_res.pdf'),  dpi=300, format='pdf')
    fig.savefig(os.path.join(current_dir, f'plots/via/via_res.jpg'),  dpi=300, format='jpg')
    fig.show()
    plt.close()

def via_plots(exps=[1, 2, 3], months = ["mar", "jun", "sep", "dec"],
              data_use=['full', 'sparse'],
              current_dir = current_dir):

    for exp in exps:
        for d in data_use:
            for month in months:
                plot_via_seasonal_together(month=month, d=d, prediction_scenario=f'exp{exp}', current_dir=current_dir)
                plot_via_seasonal_together(month=month, d=d, prediction_scenario=f'exp{exp}', current_dir=current_dir,
                                   models=['emb', 'emb_preles', 'preles'],
                                   colors=['gray', 'orange', '#efe645'],
                                   plots="emb"
                                   )

    plot_via_biascorrection(current_dir=current_dir)


def plot_predictions(obs, y_preles, y_mlp, y_pgnn, model , month = None, legend = True, save_to=''):


    if model == 'Embedded':
        print('Embedded')
        print('shorten obs')
        obs = obs[:365]
        y_mlp = y_mlp[:365]
        y_preles = y_preles[:365]
        y_pgnn = y_pgnn[:365]


    x = np.arange(len(obs))

    y_mlp_m = y_mlp.mean(axis=1)
    y_mlp_q = [np.min(y_mlp, axis=1), np.max(y_mlp, axis=1)]

    error_mlp = np.subtract(obs.to_numpy().squeeze(), y_mlp_m.to_numpy())
    error_preles = np.subtract(obs.to_numpy().squeeze(), y_preles.to_numpy().squeeze())


    plt.rc('font', **{'family': 'sans-serif'})
    plt.rcParams.update({'font.size': 26})
    fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 1]}, figsize=(8, 7))
    figlegend = plt.figure(figsize=(3, 3))

    if month == 'sep':
        x_positions = np.where(obs.index.isin(obs['2008-09-13':'2008-09-28'].index))[0]
        axs[0].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
        axs[1].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
    elif month == 'dec':
        x_positions = np.where(obs.index.isin(obs['2008-12-14':'2008-12-28'].index))[0]
        axs[0].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
        axs[1].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
    elif month == 'mar':
        x_positions = np.where(obs.index.isin(obs['2008-03-14':'2008-03-28'].index))[0]
        axs[0].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
        axs[1].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
    elif month == 'jun':
        x_positions = np.where(obs.index.isin(obs['2008-06-14':'2008-06-28'].index))[0]
        axs[0].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)
        axs[1].axvspan(x_positions.min(), x_positions.max() + 1, facecolor='lightgray', alpha=0.5)

    axs[0].fill_between(x, y_mlp_q[0], y_mlp_q[1], color='#e935a1', alpha=0.7)
    axs[0].plot(x, y_mlp_m, color='#e935a1', label='MLP', alpha = 0.75, linewidth=1.2)
    if y_pgnn is not None:
        y_pgnn_m = y_pgnn.mean(axis=1)
        y_pgnn_q = [np.min(y_pgnn, axis=1), np.max(y_pgnn, axis=1)]
        error_pgnn = np.subtract(obs.to_numpy().squeeze(), y_pgnn_m.to_numpy())
        axs[0].fill_between(x, y_pgnn_q[0], y_pgnn_q[1], color="blue", alpha=0.7)
        axs[0].plot(x, y_pgnn_m, color='blue', label='PiNN', alpha=0.8, linewidth=0.9)
    axs[0].plot(x, obs.to_numpy().squeeze(), color='black', label='Observed', linewidth=1.2, alpha=0.75)
    axs[0].plot(x, y_preles, color='yellow', label='Preles', linewidth=1.2, alpha=0.99)
    axs[0].axhline(y=0, color='black', linestyle="--", linewidth=0.9)
    if legend:
        axs[0].legend(loc='upper left', bbox_to_anchor=(0.95, 0.95),fontsize=26)
    axs[0].set_ylabel('GPP [g C m-2 day-1]')
    handles, labels = axs[0].get_legend_handles_labels()
    figlegend.legend(handles, labels, loc='center',fontsize=26)

    if y_pgnn is not None:
        axs[1].plot(x, error_pgnn, color="blue", linewidth=1.2)
    else:
        axs[1].plot(x, error_preles, color="yellow", linewidth=1.2)
    axs[1].plot(x, error_mlp, color='#e935a1', linewidth=1.2)
    axs[1].axhline(y=0, color='black', linestyle="--", linewidth=0.9)
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Time [days]')

    plt.tight_layout()
    fig.savefig(os.path.join('../plots/temporalprediction', f'{save_to}.pdf'), bbox_inches = 'tight', dpi=200, format='pdf')
    figlegend.savefig(os.path.join(current_dir, f'plots/temporalprediction/temporalprediction_legend.pdf'), dpi=300, format='pdf')
    fig.savefig(os.path.join('../plots/temporalprediction',f'{save_to}.jpg'), bbox_inches = 'tight', dpi=200, format='jpg')
    plt.close()

def prediction_plots(exps=[1, 2, 3],
                         mods=['res', 'res2', 'reg', 'mlpDA1', 'emb'],
                         months=['sep', 'dec', 'mar', 'jun']):

    for exp in exps:

        print('Plot predictions: Experiment ', exp)
        mods_names = ['Bias\n Correction', 'Parallel\nPhysics', 'Regularized\nPhysics', 'Domain\nAdaptation', 'Embedded']
        i = 0

        for mod in mods:

            print('Plot predictions: Model ', mod)
            preds_mlp, preds_mlp_sparse, preds_pgnn, preds_pgnn_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
                    exp, mod)

            for month in months:
                print('Plot predictions: Month ', month)
                plot_predictions(obs, preds_preles, preds_mlp, preds_pgnn, model=mods_names[i],
                                     save_to=f'predictions_exp{exp}_full_{mod}_{month}', month=month, legend=False)
                plot_predictions(obs, preds_preles_sparse, preds_mlp_sparse, preds_pgnn_sparse, model=mods_names[i],
                                 save_to=f'predictions_exp{exp}_sparse_{mod}_{month}', month=month, legend=False)

            i += 1

def plot_prediction_correlations(y_pgnn_full, y_pgnn_sparse, obs, save_to=''):

    """
    Example usage: plot_prediction_correlations(preds_pgnn, preds_pgnn_sparse, obs,
                                             save_to=f'predictions_correlations_exp{exp}_{mod}')
    Args:
        y_pgnn_full:
        y_pgnn_sparse:
        obs:
        save_to:

    Returns:

    """

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


def plot_prediction_correlations_together(y_pgnn, obs, colors =['#efe645','#e935a1','#00e3ff','#e1562c','#537eff', '#5954d6', 'gray'],
                                  legend = True, save_to=''):


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
    markers = ['o', 's', 'v', 'D', 'P', 'X', '^']
    labels = ['PRELES', 'MLP', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regularised\nPhysics', 'Domain\nAdaptation', 'Embedding']

    plt.plot(x, y, linestyle = '--', color = 'black')
    for i in range(len(means)):
        plt.errorbar(bins[:-1], means[i], yerr=stds[i], linestyle='-', marker=markers[i],xuplims=True, xlolims=True,
                         markersize = 13, alpha=0.89, color=colors[i], label=labels[i])

    plt.ylabel("Predicted GPP [g C m-2 day-1]")
    plt.xlabel("Observed GPP [g C m-2 day-1]")

    if legend:
        plt.legend(loc='upper left', fontsize=20, ncol= 2, handleheight=2.4, labelspacing=0.07) #loc='upper left', bbox_to_anchor=(1, 1)
    plt.tight_layout()
    plt.savefig(os.path.join('../plots/correlation', f'{save_to}.pdf'), bbox_inches = 'tight', dpi=200, format='pdf')
    plt.savefig(os.path.join('../plots/correlation', f'{save_to}.jpg'), bbox_inches='tight', dpi=200, format='jpg')
    plt.close()

def correlation_plots(exps = [1, 2, 3]):

    for exp in exps:

        preds_mlp, preds_mlp_sparse, preds_res, preds_res_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp = exp, mod = 'res')
        preds_mlp, preds_mlp_sparse, preds_res2, preds_res2_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp = exp, mod = 'res2')
        preds_mlp, preds_mlp_sparse, preds_reg, preds_reg_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp=exp, mod='reg')
        preds_mlp, preds_mlp_sparse, preds_mlpDA1, preds_mlpDA1_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp=exp, mod='mlpDA1')
        preds_mlp, preds_mlp_sparse, preds_emb, preds_emb_sparse, obs, preds_preles, preds_preles_sparse = get_predictions(
            exp=exp, mod='emb')
        if exp == 1:
            preds_emb, preds_emb_sparse = preds_emb[:365], preds_emb_sparse[:365]


        assembled_preds_full = [preds_preles, preds_mlp, preds_res,preds_res2, preds_reg, preds_mlpDA1, preds_emb]
        assembled_preds_sparse = [preds_preles_sparse, preds_mlp_sparse, preds_res_sparse,preds_res2_sparse, preds_reg_sparse, preds_mlpDA1_sparse, preds_emb_sparse]

        plot_prediction_correlations_together(assembled_preds_full, obs, legend = False,
                                        save_to=f'predictions_correlations_exp{exp}_all_full')
        plot_prediction_correlations_together(assembled_preds_sparse, obs, legend =True,
                                        save_to=f'predictions_correlations_exp{exp}_all_sparse')


if __name__ == '__main__':


    performance_plots()

    via_plots()

    prediction_plots()

    correlation_plots()

