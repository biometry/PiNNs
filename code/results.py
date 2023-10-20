#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:47:12 2022

@author: Marieke_Wesselkamp
"""

import sys, os
sys.path.append("/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn/code")
os.chdir("/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn/code")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import get_seasonal_data

current_dir = '/Users/Marieke_Wesselkamp/PycharmProjects/physics_guided_nn'

#%%
def plot_performance(performance, prediction, data_use, log=False, save=True):

    plt.rc('font',**{'family':'sans-serif'})
    
    fig,ax = plt.subplots(figsize=(8,8))
    #ax = fig.add_axes([1,1,1,1])
    # Creating plot
    if log:
        bp = ax.boxplot(np.log(performance), patch_artist=True)
    else:
        bp = ax.boxplot(performance, patch_artist=True)
        
    colors = ['#d7191c','#fdae61','#ffffbf','#abd9e9','#2c7bb6', '#998ec3']
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
    #if data_use == 'sparse':
    #
    #else: 
    #    ax.set_xticklabels(['', '', '', '', '', '']) 
    
    if not log:
        if (prediction == 'temporal') & (data_use == 'full'):
            ax.set_ylim(0.5,1.3)
        if (prediction == 'temporal') & (data_use == 'sparse'):
            ax.set_ylim(0.5,1.3)
        if (prediction == 'spatial') & (data_use == 'full'):
            ax.set_ylim(0.5,4.0)
        if (prediction == 'spatial') & (data_use == 'sparse'):
            ax.set_ylim(0.5,4.0)
    
    if log:
        ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=20)
    else:
        ax.set_ylabel('Mean absolute error [g C m$^{-2}$ day$^{-1}$]', fontsize=24)
        
    if prediction == 'temporal':
        ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrised', 'Domain\nAdaptation'],
                       rotation = 45)
    else:
        ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrised', 'Domain\nAdaptation'],
                       rotation = 45)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(22)
        #tick.label.set_fontfamily({'font.serif':'Palatino'}) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(22)
        #tick.label.set_fontfamily('Palatino Linotype')
    plt.tight_layout()
    plt.show()
    
    if save:
    
        if log:
            fig.savefig(os.path.join(current_dir, f"plots/performance_{prediction}_{data_use}_log.pdf"), bbox_inches = 'tight', dpi=200, format='pdf')
        else:
            fig.savefig(os.path.join(current_dir, f"plots/performance_{prediction}_{data_use}.pdf"), bbox_inches = 'tight', dpi=200, format='pdf')


#%%
def plot_performance_summarized(dat1, dat2, prediction):

    plt.rc('font',**{'family':'sans-serif','sans-serif':['Fira Sans OT']})
    
    def draw_plot(data, offset, edgecolor, colors):
        pos = np.arange(data.shape[1])+offset
        bp = ax.boxplot(data, positions = pos, widths = 0.3, patch_artist=True)
    
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
        
    fig,ax = plt.subplots(figsize=(8,7), dpi=200)
    #ax = fig.add_axes([0,0,1,1])
    if prediction == 'temp':
        colors1 = ['#2c7bb6','#2c7bb6','#2c7bb6','#2c7bb6','#2c7bb6']
        colors2 = ['#abd9e9','#abd9e9','#abd9e9','#abd9e9','#abd9e9']
        red_patch = mpatches.Patch(color='#2c7bb6', label='Full')
        blue_patch = mpatches.Patch(color='#abd9e9', label='Sparse')
    else:
        colors1 = ['#fdae61','#fdae61','#fdae61','#fdae61','#fdae61']
        colors2 = ['#d7191c','#d7191c','#d7191c','#d7191c','#d7191c']
        red_patch = mpatches.Patch(color='#fdae61', label='Full')
        blue_patch = mpatches.Patch(color='#d7191c', label='Sparse')
        
    draw_plot(dat1, -0.2, edgecolor="black", colors=colors1)
    draw_plot(dat2, +0.2, edgecolor="black", colors=colors2)
    
    plt.xticks(np.arange(5))
    ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrized', 'Domain\nAdaptation']) #, 'Domain\nAdaptation'])
    ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=20)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20) 
        #tick.label.set_fontfamily({'font.serif':'Palatino'}) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        #tick.label.set_fontfamily('Palatino Linotype') 
    #ax = fig.add_axes([0,0,1,1])
    if prediction == 'temp':
        plt.legend(handles=[red_patch, blue_patch], fontsize=18, loc="upper right")
    else:
        plt.legend(handles=[red_patch, blue_patch], fontsize=18, loc="lower left")
    
    plt.show()
    fig.savefig(f'../plots/performance_{prediction}_summarized.png', bbox_inches = 'tight', dpi=200)

#plot_performance_summarized(np.log(performances_all[0]), np.log(performances_all[2]), prediction='temp')
#plot_performance_summarized(np.log(performances_all[1]), np.log(performances_all[3]), prediction='spat')

#%% VIA conditional plots

def observed_seasonal_means(data_use = 'full', model = 'mlp', prediction_scenario='temporal'):

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

def plot_via(d = "full", prediction_scenario = 'spatial', current_dir ='', save=True):

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
            vi3 = np.array(pd.read_csv(os.path.join(current_dir,f"results_final/via/{prediction_scenario}/{mod}_{d}_{v}_via_cond_{months[i]}.csv"), index_col=False).iloc[:,1:])
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
        fig.savefig(os.path.join(current_dir, f'plots/via_{prediction_scenario}_{d}.pdf'),  dpi=300, format='pdf')

    fig.show()

def plot_via_biascorrection(current_dir =''):

    months = ["dec", "mar", "jun", "sep"]
    days = ["Spring", "Summer", "Autum", "Winter"]
    colors = ["lightblue","darkgreen",  "lightgreen", "darkblue"]
    var = ["$GPP$ [g C m$^{-2}$ day$^{-1}$]", "$GPP$ [g C m$^{-2}$ day$^{-1}$]", "$GPP$ [g C m$^{-2}$ day$^{-1}$]", "$GPP$ [g C m$^{-2}$ day$^{-1}$]",
           "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]", "$ET$ [mm]",
           "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]", "$SW$ [mm]"]

    cols = ["Temporal full", "Temporal sparse", "Spatial full", "Spatial sparse"]

    fig = plt.figure(figsize=(80,12))
    widths = [i for i in np.repeat(3, 4)]
    heights = [i for i in np.repeat(3, 3)]

    gs = fig.add_gridspec(3, 4, width_ratios = widths, height_ratios=heights, wspace=0.5, hspace=0.8)
    (ax1, ax2, ax3, ax4) , \
    (ax5, ax6, ax7, ax8) , \
    (ax9, ax10, ax11, ax12)  = gs.subplots()

    all_means, var_ranges, mn, std = observed_seasonal_means(model='res')

    def plot_variable(v,ax, mod, d, prediction_scenario):
        for i in range(4):
            all_means, var_ranges, mn, std = observed_seasonal_means(data_use=d,model='res',
                                                                     prediction_scenario=prediction_scenario)

            vi3 = np.array(pd.read_csv(os.path.join(current_dir,f"results_final/via/{prediction_scenario}/{mod}_{d}_{v}_via_cond_{months[i]}.csv"), index_col=False).iloc[:,1:])
            vi3_m = vi3.mean(axis=1)
            vi3_q = np.quantile(vi3, (0.05, 0.95), axis=1)
            ax.fill_between(var_ranges[v], vi3_q[0],vi3_q[1],color=colors[i], alpha=0.2)
            ax.plot(var_ranges[v], vi3_m, color=colors[i], label=days[i])
            ax.plot(all_means[v][i], all_means['GPP'][i], marker='x', markersize=10, color=colors[i])

    j=0
    axs = (ax1, ax5, ax9)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'full', 'temporal')
        j += 1

    j=0
    axs = (ax2, ax6, ax10)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'sparse', 'temporal')
        j += 1

    j=0
    axs = (ax3, ax7, ax11)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'full', 'spatial')
        j += 1

    j=0
    axs = (ax4, ax8, ax12)
    for key, value in var_ranges.items():
        plot_variable(key, axs[j],'res', 'sparse', 'spatial')
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

    fig.savefig(os.path.join(current_dir, f'plots/via_res.pdf'),  dpi=300, format='pdf')
    fig.show()


if __name__ == '__main__':

    mods = ['preles', 'mlp', 'res', 'res2', 'reg', 'mlpDA1']  # , "emb"
    data_use = ['full', 'sparse']
    perf = np.zeros((4, len(mods)))
    perf2 = np.zeros((5, len(mods)))
    performances_all = []
    for d in data_use:
        i = 0
        for mod in mods:
            if (mod == 'mlpDA1' or mod == 'mlpDA2'):
                perf[:, i] = pd.read_csv(f"../results_final/temporal/{d}/{mod}_eval_performance_{d}.csv",
                                         index_col=False).iloc[:, 4]
                perf2[:, i] = pd.read_csv(f"../results_final/spatial/{d}/2{mod}_eval_performance_{d}.csv",
                                          index_col=False).iloc[:, 4]
            elif mod != 'preles':
                perf[:, i] = pd.read_csv(f"../results_final/temporal/{d}/{mod}_eval_{d}_performance.csv",
                                         index_col=False).iloc[:, 4]
                perf2[:, i] = pd.read_csv(f"../results_final/spatial/{d}/2{mod}_eval_{d}_performance.csv",
                                          index_col=False).iloc[:, 4]
            else:
                perf[:, i] = pd.read_csv(f"../results_final/temporal/{d}/{mod}_eval_{d}_performance.csv",
                                         index_col=False).iloc[:, 2]
                perf2[:, i] = pd.read_csv(f"../results_final/spatial/{d}/2{mod}_eval_{d}_performance.csv",
                                          index_col=False).iloc[:, 2]

            i += 1
        performances_all.append(perf.copy())
        performances_all.append(perf2.copy())
        plot_performance(perf, 'temporal', d, log=False, save=True)
        plot_performance(perf2, 'spatial', d, log=False, save=True)
        print(f'Mean temp prediction performance {d}:',
              list(zip(np.round(np.mean(perf, axis=0), 2), np.round(np.std(perf, axis=0), 2))))
        print(f'Mean spat prediction performance {d}:',
              list(zip(np.round(np.mean(perf2, axis=0), 2), np.round(np.std(perf2, axis=0), 2))))

    plot_via(d="full", prediction_scenario='temporal', current_dir=current_dir, save=True)
    plot_via(d="sparse", prediction_scenario='temporal', current_dir=current_dir, save=True)
    plot_via(d="full", prediction_scenario='spatial', current_dir=current_dir, save=True)
    plot_via(d="sparse", prediction_scenario='spatial', current_dir=current_dir, save=True)

    plot_via_biascorrection(current_dir=current_dir)