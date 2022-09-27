#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 08:47:12 2022

@author: Marieke_Wesselkamp
"""

import os
os.chdir("/Users/Marieke_Wesselkamp/Projects/physics_guided_nn/code")

import pandas as pd
import numpy as np
import utils 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#%%
def plot_performance(perf, prediction, data_use, log=False, save=True):

    plt.rc('font',**{'family':'sans-serif','sans-serif':['Fira Sans OT']})
    
    fig = plt.figure(figsize=(7,7), dpi=200)
    ax = fig.add_axes([0,0,1,1])
    # Creating plot
    if log:
        bp = ax.boxplot(np.log(perf), patch_artist=True)
    else:
        bp = ax.boxplot(perf, patch_artist=True)
        
    colors = ['#d7191c','#fdae61','#ffffbf','#abd9e9','#2c7bb6', '#998ec3', 'blue']
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
        if (prediction == 'temp') & (data_use == 'full'):
            ax.set_ylim(0.5,3.5)
        if (prediction == 'temp') & (data_use == 'sparse'):
            ax.set_ylim(0.5,3.5)
        if (prediction == 'spat') & (data_use == 'full'):
            ax.set_ylim(0.5,5.5)
        if (prediction == 'spat') & (data_use == 'sparse'):
            ax.set_ylim(0.5,5.5)
    
    if log:
        ax.set_ylabel('$log(\mathrm{GPP} - \widehat{\mathrm{GPP}})$ [g C m$^{-2}$ day$^{-1}$]', fontsize=20)
    else:
        ax.set_ylabel('Mean absolute error [g C m$^{-2}$ day$^{-1}$]', fontsize=24)
        
    if prediction == 'temp':
        ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrised', 'Domain\nAdaptation', 'Embedded'],
                       rotation = 45)
    else:
        ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrised', 'Domain\nAdaptation', 'Embedded'],
                       rotation = 45)
        
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(22) 
        #tick.label.set_fontfamily({'font.serif':'Palatino'}) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(22) 
        #tick.label.set_fontfamily('Palatino Linotype') 

    plt.show()
    
    if save:
    
        if log:
            fig.savefig(f'../plots/performance_{prediction}_{data_use}_log.pdf', bbox_inches = 'tight', dpi=200, format='pdf')
        else:
            fig.savefig(f'../plots/performance_{prediction}_{data_use}.pdf', bbox_inches = 'tight', dpi=200, format='pdf')
        
#%%
mods = ['preles','mlp', 'res', 'res2', 'reg', 'mlpDA1', "emb"]
data_use = ['full', 'sparse']

perf = np.zeros((4,len(mods)))
perf2 = np.zeros((5,len(mods)))

performances_all = []

for d in data_use:
    i=0
    for mod in mods:
    
        if (mod == 'mlpDA1' or mod == 'mlpDA2' or mod == 'emb'):
            perf[:,i] = pd.read_csv(f"../results/{mod}_eval_performance_{d}.csv", index_col=False ).iloc[:,4]
            if mod != 'emb':
                perf2[:,i] = pd.read_csv(f"../results/2{mod}_eval_performance_{d}.csv", index_col=False ).iloc[:,4]
        elif mod != 'preles':
            perf[:,i] = pd.read_csv(f"../resultsN2/{mod}_eval_{d}_performance.csv", index_col=False ).iloc[:,4]
            perf2[:,i] = pd.read_csv(f"../resultsN2/2{mod}_eval_{d}_performance.csv", index_col=False ).iloc[:,4]
        else:
            perf[:,i] = pd.read_csv(f"../resultsN2/{mod}_eval_{d}_performance.csv", index_col=False ).iloc[:,2]
            perf2[:,i] = pd.read_csv(f"../resultsN2/2{mod}_eval_{d}_performance.csv", index_col=False ).iloc[:,2]
        
        i += 1
    performances_all.append(perf.copy())
    performances_all.append(perf2.copy())
    
    plot_performance(perf, 'temp', d, save=True)
    plot_performance(perf2, 'spat', d, save=True)
    
    print(f'Mean temp prediction performance {d}:', list(zip(np.round(np.mean(perf, axis=0), 2), np.round(np.std(perf, axis=0), 2))))
    print(f'Mean spat prediction performance {d}:', list(zip(np.round(np.mean(perf2, axis=0), 2), np.round(np.std(perf2, axis=0), 2))))

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
    ax.set_xticklabels(['$\mathbf{Preles}$', '$\mathbf{Naive}$', 'Bias\nCorrection', 'Parallel\nPhysics', 'Regula-\nrized']) #, 'Domain\nAdaptation'])      
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
#%%
plot_performance_summarized(np.log(fuckyoulist[0]), np.log(fuckyoulist[2]), prediction='temp')
plot_performance_summarized(np.log(fuckyoulist[1]), np.log(fuckyoulist[3]), prediction='spat')

#%%
def plot_via(vi, labels, var, vi2 = None, vi3 = None, main = None, xticks = None, xlabel = None, legend=True):
    
    vi = vi.iloc[:,1:]
    vi2 = vi2.iloc[:,1:]
    
    vm = np.mean(vi, axis=1) 
    vu = vm + 2*np.std(vi, axis=1)
    vl = vm - 2*np.std(vi, axis=1)
        
    fig, ax = plt.subplots(figsize=(7,7))
    #ax.set_ylim((,10))
    ax.fill_between(np.arange(len(vm)), vu, vl, color='#fdae61', alpha = 0.5, label = labels[0])
    ax.plot(vm, color="black")
    
    if not vi2 is None:
        vm = np.mean(vi2, axis=1) 
        vu = vm + 2*np.std(vi2, axis=1)
        vl = vm - 2*np.std(vi2, axis=1)
        
        ax.fill_between(np.arange(len(vm)), vu, vl, color='#2c7bb6', alpha=0.5, label=labels[1])
        ax.plot(vm, color="black")
        
    if not vi3 is None:
        vm = np.mean(vi3.iloc, axis=1) 
        vu = vm + 2*np.std(vi3, axis=1)
        vl = vm - 2*np.std(vi3, axis=1)
        
        ax.fill_between(np.arange(len(vm)), vu, vl, color='#abd9e9', alpha=0.5, label=labels[2])
        ax.plot(vm, color="black")
    
    if not xticks is None:
        if var =='Tair':
            ax.xaxis.set_ticks([0,33,66, 100, 133, 166, 200])
        elif var == 'PAR':
            ax.xaxis.set_ticks([0, 25, 50, 75, 100, 125, 150, 175, 200])
        elif var == 'Precip':
            ax.xaxis.set_ticks([0,  20,  40,  60,  80, 100, 120, 140, 160, 180, 200])
        ax.set_xticklabels(xticks)
    
    if not main is None:
        ax.set_title(main, fontsize = 24, y=1.08)    
        
    for tick in ax.yaxis.get_major_ticks():
        print(tick)
        tick.label.set_fontsize(18) 
        #tick.label.set_fontfamily({'font.serif':'Palatino'}) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(18)
        
    if legend:
        ax.legend(fontsize=18, loc=(0.7,1.02))
    ax.set_ylabel("Predicted yearly GPP [g C m$^{-2}$ day$^{-1}$]", fontsize=18)
    
    if not xlabel is None:
        ax.set_xlabel(xlabel, fontsize=18)
        
    if legend:
        fig.savefig(f'../plots/VIA_{var}.pdf', bbox_inches = 'tight', dpi=300, format='pdf')
    else:
        fig.savefig(f'../plots/VIA_{var}2.pdf', bbox_inches = 'tight', dpi=300)
        
#%% 
mods = ['preles','mlp', 'res', 'res2', 'reg']
data_use = ['full'] #, 'sparse']
variables = ['VPD', 'Precip', 'Tair', 'fapar', 'PAR']
xlabels = ['Vapor pressure deficit [kPA]', 'Precipiation [mm]', 'Mean air temperature [$^{\circ}$C]', 
           'Fraction of photosynthetic active radiation', 'Photosynthetic active radiation [mol m$^{-2}$ d$^{-1}$]']

var = 'Tair'
mod = 'reg'
d = 'full'

i = 0
for var in variables:
    for d in data_use:
        vi = pd.read_csv(f"../resultsN2/mlp_{d}_{var}_via.csv", index_col=False, header=1)
        vi1 = pd.read_csv(f"../resultsN2/reg_{d}_{var}_via.csv", index_col=False, header=1).iloc[:,1:]
        vi2 = pd.read_csv(f"../resultsN2/res2_{d}_{var}_via.csv", index_col=False, header=1).iloc[:,1:]
        vi3 = pd.read_csv(f"../results/preles_{d}_{var}_via.csv", index_col=False, header=1).iloc[:,1:]
        if var=='fapar':
            xticks = [0] + list(np.round(vi.iloc[0::24,0],1))
        elif var == 'Tair':
            xticks = [-20, -10, 0, 10, 20, 30, 40]
        elif var == 'PAR':
            xticks = [0, 25, 50, 75, 100, 125, 150, 175, 200]
        elif var== 'Precip':
            xticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        else:
            xticks = [0] + [int(x) for x in list(np.rint(vi.iloc[0::25,0]))] + [int(vi.iloc[-1,0])]
            
        plot_via(vi, labels=['Naive', 'Preles', 'Parallel Physics'], 
                 var = var,
                 vi2 = vi3,vi3=vi2, main = var, 
                 xticks = xticks ,
                 xlabel=xlabels[i],
                 legend = True)
    i += 1



#%%plt
mod='mlp'
d = 'sparse'

for var in variables:
        vi = pd.read_csv(f"../results/{mod}_{d}_{var}_via.csv", index_col=False, header=1).iloc[:,1:]
        plot_via(vi)
#%% VIA sumamrized
fig, ax = plt.subplots(figsize=(7,7))
i = 0
n = ['Preles', 'Parallel Physics', 'Naive']
colors = ['#2c7bb6', '#abd9e9', '#fdae61']
marker = ['.', '^', '*', '+', 'p']

for var in variables:
    vias = np.zeros((3,2))
    for d in data_use:
        vi = pd.read_csv(f"../results/preles_{d}_{var}_via.csv", index_col=False, header=1)#.iloc[:,1:]
        vi2 = pd.read_csv(f"../resultsN2/res2_{d}_{var}_via.csv", index_col=False, header=1)#.iloc[:,1:]
        vi3 = pd.read_csv(f"../resultsN2/mlp_{d}_{var}_via.csv", index_col=False, header=1)#.iloc[:,1:]
        
        vm = np.mean(vi.iloc[:,1:], axis=1) 
        print('Preles')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[0,0] = np.std(vm)
    
        vm = np.mean(vi.iloc[:,1:], axis=0) 
        print('Preles')
        print("Yeary standard dev in GPP predictions at values of ", var)
        print(np.std(vm))
        vias[0,1] = np.std(vm)
        
        vm = np.mean(vi2.iloc[:,1:], axis=1) 
        print('Parallel Physics')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[1,0] = np.std(vm)
        
        vm = np.mean(vi2.iloc[:,1:], axis=1)
        print('Parallel Physics')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[1,1] = np.std(vm)
        
        vm = np.mean(vi3.iloc[:,1:], axis=1) 
        print('Naive')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[2,0] = np.std(vm)
        
        vm = np.mean(vi3.iloc[:,1:], axis=1)
        print('Naive')
        print("Standard dev in yearly GPP predictions over values of ", var)
        print(np.std(vm))
        vias[2,1] = np.std(vm)
        
        
    ax.scatter(vias[0,1], vias[0,0], label = var, color='#fdae61', marker = marker[i], s=300)
    ax.scatter(vias[1,1], vias[1,0], label = var, color='#abd9e9', marker = marker[i], s=300)
    ax.scatter(vias[2,1], vias[2,0], label = var, color='#2c7bb6', marker = marker[i], s=300)
#    for i, txt in enumerate(n):
#        ax.annotate(txt, (vias[:,1][i]+0.1, vias[:,0][i]+0.1), size=12, color="black")
            
    i += 1
    
ax.set_xlabel("$\sigma(\overline{\widehat{\mathrm{GPP}}})$", fontsize=18)
ax.set_ylabel("$\overline{\sigma(\widehat{\mathrm{GPP}}_{i})}$", fontsize=18)

# legend
label_column = variables
label_row = n
color = np.array([colors,]*5).transpose()
rows = [mpatches.Patch(color=color[i, 0]) for i in range(3)]
columns = [plt.plot([], [], marker[i], markerfacecolor='w',
                    markeredgecolor='k', markersize=12)[0] for i in range(5)]

plt.legend(rows + columns, label_row + label_column, fontsize = 12, loc='lower right')


for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(18) 
    #tick.label.set_fontfamily({'font.serif':'Palatino'}) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(18)

fig.savefig(f'../plots/VIA_summarized.pdf', bbox_inches = 'tight', dpi=300, format='pdf')

#%% VIA conditional plots I
d = "full"
var = ["TAir [$\degree$C]", "TAir [$\degree$C]", "TAir [$\degree$C]", "VPD [kPa]", "VPD [kPa]", "VPD [kPa]", 
       "Precip [mm]", "Precip [mm]", "Precip [mm]", "PAR [mol m$^{-2}$ d$^{-1}$]", "PAR [mol m$^{-2}$ d$^{-1}$]", "PAR [mol m$^{-2}$ d$^{-1}$]",
       "fapar", "fapar", "fapar"]
ylabels = ["Conditional GPP Predictions", "", "", 
           "Conditional GPP Predictions", "", "", 
           "Conditional GPP Predictions", "", "",
           "Conditional GPP Predictions", "", "",
           "Conditional GPP Predictions", "", ""]

gridsize=200
Tair_range = np.linspace(-20, 40, gridsize)
VPD_range = np.linspace(0, 60, gridsize)
Precip_range = np.linspace(0, 100, gridsize)
PAR_range = np.linspace(-20, 40, gridsize)
fapar_range = np.linspace(0, 1, gridsize)
days = ["Spring", "Summer", "Autum", "Winter"]
colors = ["lightgreen", "lightblue", "green", "darkblue"]
cols = ["Preles", "Naive Network", "Parallel Physics Network"]

vi3 = np.array(pd.read_csv(f"../results/preles_{d}_Tair_via_conditional.csv", index_col=False).iloc[:,1:])
vi2 = np.transpose(np.array(pd.read_csv(f"../results/mlp_{d}_Tair_via_conditional.csv", index_col=False).iloc[:,1:]))
vi1 = np.transpose(np.array(pd.read_csv(f"../results/res2_{d}_Tair_via_conditional.csv", index_col=False).iloc[:,1:]))
vi4 = np.array(pd.read_csv(f"../results/preles_{d}_VPD_via_conditional.csv", index_col=False).iloc[:,1:])
vi5 = np.transpose(np.array(pd.read_csv(f"../results/mlp_{d}_VPD_via_conditional.csv", index_col=False).iloc[:,1:]))
vi6 = np.transpose(np.array(pd.read_csv(f"../results/res2_{d}_VPD_via_conditional.csv", index_col=False).iloc[:,1:]))
vi7 = np.array(pd.read_csv(f"../results/preles_{d}_Precip_via_conditional.csv", index_col=False).iloc[:,1:])
vi8 = np.transpose(np.array(pd.read_csv(f"../results/mlp_{d}_Precip_via_conditional.csv", index_col=False).iloc[:,1:]))
vi9 = np.transpose(np.array(pd.read_csv(f"../results/res2_{d}_Precip_via_conditional.csv", index_col=False).iloc[:,1:]))
vi10 = np.array(pd.read_csv(f"../results/preles_{d}_PAR_via_conditional.csv", index_col=False).iloc[:,1:])
vi11 = np.transpose(np.array(pd.read_csv(f"../results/mlp_{d}_PAR_via_conditional.csv", index_col=False).iloc[:,1:]))
vi12 = np.transpose(np.array(pd.read_csv(f"../results/res2_{d}_PAR_via_conditional.csv", index_col=False).iloc[:,1:]))
vi13 = np.array(pd.read_csv(f"../results/preles_{d}_fapar_via_conditional.csv", index_col=False).iloc[:,1:])
vi14 = np.transpose(np.array(pd.read_csv(f"../results/mlp_{d}_fapar_via_conditional.csv", index_col=False).iloc[:,1:]))
vi15 = np.transpose(np.array(pd.read_csv(f"../results/res2_{d}_fapar_via_conditional.csv", index_col=False).iloc[:,1:]))


fig = plt.figure(figsize=(21,35))

gs = fig.add_gridspec(5, 3, wspace=0.05, hspace=0.2)
(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15) = gs.subplots(sharey='row')


for i in range(len(days)):
    ax1.plot(Tair_range, vi3[:,i], color=colors[i], label=days[i])
for i in range(len(days)):
    ax2.plot(Tair_range, np.transpose(vi2[:,i]), color=colors[i], label=days[i])
for i in range(len(days)):
    ax3.plot(Tair_range, vi1[:,i], color=colors[i], label=days[i])
    
for i in range(len(days)):
    ax4.plot(VPD_range, vi4[:,i], color=colors[i], label=days[i])
for i in range(len(days)):
    ax5.plot(VPD_range, np.transpose(vi5[:,i]), color=colors[i], label=days[i])
for i in range(len(days)):
    ax6.plot(VPD_range, vi6[:,i], color=colors[i], label=days[i])
    
for i in range(len(days)):
    ax7.plot(Precip_range, vi7[:,i], color=colors[i], label=days[i])
for i in range(len(days)):
    ax8.plot(Precip_range, np.transpose(vi8[:,i]), color=colors[i], label=days[i])
for i in range(len(days)):
    ax9.plot(Precip_range, vi9[:,i], color=colors[i], label=days[i])
    
for i in range(len(days)):
    ax10.plot(Precip_range, vi10[:,i], color=colors[i], label=days[i])
for i in range(len(days)):
    ax11.plot(Precip_range, np.transpose(vi11[:,i]), color=colors[i], label=days[i])
for i in range(len(days)):
    ax12.plot(Precip_range, vi12[:,i], color=colors[i], label=days[i])
    
for i in range(len(days)):
    ax13.plot(Precip_range, vi13[:,i], color=colors[i], label=days[i])
for i in range(len(days)):
    ax14.plot(Precip_range, np.transpose(vi14[:,i]), color=colors[i], label=days[i])
for i in range(len(days)):
    ax15.plot(Precip_range, vi15[:,i], color=colors[i], label=days[i])
    

i = 0
for ax in fig.get_axes():
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel(f"Range of {var[i]}", size=20)
    #ax.set_ylabel(f"{ylabels[i]}", size=20)
    i += 1
    #ax.label_outer()
    
axs = fig.get_axes()
for i in range(3):
    axs[i].set_title(cols[i], size=24)
    
fig.text(0.07, 0.5, 'Conditional GPP Predictions [g C m$^{-2}$ day$^{-1}$]', va='center', rotation='vertical', size=20) 
  
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc = (0.3, 0.048),ncol=4, fontsize=20) # loc=(0., 0.05)

fig.savefig(f'../plots/VIA_conditional.pdf', bbox_inches = 'tight', dpi=300, format='pdf')
fig.show()


#%% VIA conditional plots II
d = "full"
months = ["dec", "mar", "jun", "sep"]
days = ["Spring", "Summer", "Autum", "Winter"]
colors = ["lightblue","darkgreen",  "lightgreen", "darkblue"]
var = ["$T$ [$\degree$C]", "$T$ [$\degree$C]", "$T$ [$\degree$C]", "$D$ [kPa]", "$D$ [kPa]", "$D$ [kPa]", 
       "$R$ [mm]", "$R$ [mm]", "$R$ [mm]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]", "$\phi$ [mol m$^{-2}$ d$^{-1}$]",
       "$f_{aPPFD}$", "$f_{aPPFD}$", "$f_{aPPFD}$"]
ylabels = ["Conditional GPP Predictions", "", "", 
           "Conditional GPP Predictions", "", "", 
           "Conditional GPP Predictions", "", "",
           "Conditional GPP Predictions", "", "",
           "Conditional GPP Predictions", "", ""]
cols = ["Process Model", "Naive Neural Network", "Parallel Physics Network"]

gridsize=200
Tair_range = np.linspace(-20, 40, gridsize)
VPD_range = np.linspace(0, 60, gridsize)
Precip_range = np.linspace(0, 100, gridsize)
PAR_range = np.linspace(-20, 40, gridsize)
fapar_range = np.linspace(0, 1, gridsize)
variables = {'TAir':Tair_range, 'VPD':VPD_range, 'Precip':Precip_range, 'PAR':PAR_range, 'fapar':fapar_range}

fig = plt.figure(figsize=(21,35))

gs = fig.add_gridspec(5, 3, wspace=0.05, hspace=0.2)
(ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15) = gs.subplots(sharey='row')

def plot_variable(v,ax, mod):
    for i in range(4):
        vi3 = np.array(pd.read_csv(f"../results/{mod}_{d}_{v}_via_cond_{months[i]}.csv", index_col=False).iloc[:,1:])
        vi3_m = vi3.mean(axis=1)
        vi3_q = np.quantile(vi3, (0.05, 0.95), axis=1)
        ax.fill_between(variables[v], vi3_q[0],vi3_q[1],color=colors[i], alpha=0.2)
        ax.plot(variables[v], vi3_m, color=colors[i], label=days[i])

j=0
axs = (ax1, ax4, ax7, ax10, ax13)
for key, value in variables.items():
    plot_variable(key, axs[j],'preles')
    j += 1
    
j=0
axs = (ax2, ax5, ax8, ax11, ax14)
for key, value in variables.items():
    plot_variable(key, axs[j],'mlp')
    j += 1
    
j=0
axs = (ax3, ax6, ax9, ax12, ax15)
for key, value in variables.items():
    plot_variable(key, axs[j],'res2')
    j += 1
    
    
i = 0
for ax in fig.get_axes():
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.set_xlabel(f"Range of {var[i]}", size=20)
    #ax.set_ylabel(f"{ylabels[i]}", size=20)
    i += 1
    #ax.label_outer()
    
axs = fig.get_axes()
for i in range(3):
    axs[i].set_title(cols[i], size=24)
    
fig.text(0.07, 0.5, 'Conditional GPP Predictions [g C m$^{-2}$ day$^{-1}$]', va='center', rotation='vertical', size=20) 
  
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc = (0.3, 0.048),ncol=4, fontsize=20) # loc=(0., 0.05)

fig.savefig(f'../plots/ice_plots.pdf', bbox_inches = 'tight', dpi=300, format='pdf')
fig.show()
