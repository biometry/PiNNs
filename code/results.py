# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import os
os.chdir("/Users/Marieke_Wesselkamp/physics_guided_nn/code")

import pandas as pd
import numpy as np
import utils 
import matplotlib
import matplotlib.pyplot as plt
#from pylab import *
from sklearn import metrics
#%%
plt.rcParams.update({
  "text.usetex": False,
  "font.family": "Helvetica"
})

#%%
x, y, xt = utils.loaddata('validation', 1, dir="../data/", raw=True)
y = y.to_frame()

#test_x = x_te[x_te.index.year == 2008]
y2008 = y[y.index.year == 2008]
y2008s = y2008.iloc[:-1,:]
#%%
performance_mlp = pd.read_csv("../results/mlp_eval_performance_full.csv", index_col=False ).iloc[:,1:5]
predictions_mlp = pd.read_csv("../results/mlp_eval_preds_test_full.csv",index_col=False ).iloc[:,1:]
tloss_mlp = pd.read_csv("../results/mlp_trainloss_full.csv",index_col=False ).iloc[:,1:]
vloss_mlp = pd.read_csv("../results/mlp_vloss_full.csv",index_col=False ).iloc[:,1:]

performance_res = pd.read_csv("../results/res_eval_performance_full.csv", index_col=False ).iloc[:,1:5]
predictions_res = pd.read_csv("../results/res_eval_preds_test_full.csv",index_col=False ).iloc[:,1:]
tloss_res = pd.read_csv("../results/res_trainloss_full.csv",index_col=False ).iloc[:,1:]
vloss_res = pd.read_csv("../results/res_vloss_full.csv",index_col=False ).iloc[:,1:]


performance_res2 = pd.read_csv("../results/res2_eval_performance_full.csv", index_col=False ).iloc[:,1:5]
predictions_res2 = pd.read_csv("../results/res2_eval_preds_test_full.csv",index_col=False ).iloc[:,1:]
tloss_res2 = pd.read_csv("../results/res2_trainloss_full.csv",index_col=False ).iloc[:,1:]
vloss_res2 = pd.read_csv("../results/res2_vloss_full.csv",index_col=False ).iloc[:,1:]


performance_reg = pd.read_csv("../results/reg_eval_performance_full.csv", index_col=False ).iloc[:,1:5]
predictions_reg = pd.read_csv("../results/reg_eval_preds_test_full.csv",index_col=False ).iloc[:,1:]
tloss_reg = pd.read_csv("../results/reg_trainloss_full.csv",index_col=False ).iloc[:,1:]
vloss_reg = pd.read_csv("../results/reg_vloss_full.csv",index_col=False ).iloc[:,1:]

performance_da1 = pd.read_csv("../results/mlpDA1_eval_performance_full.csv", index_col=False ).iloc[:,1:5]
predictions_da1 = pd.read_csv("../results/mlpDA1_eval_preds_test_full.csv",index_col=False ).iloc[:,1:]
tloss_da1 = pd.read_csv("../results/mlpDA1_trainloss_full.csv",index_col=False ).iloc[:,1:]
vloss_da1 = pd.read_csv("../results/mlpDA1_vloss_full.csv",index_col=False ).iloc[:,1:]

#performance_da2 = pd.read_csv("../results/mlpDA2_eval_performance_full.csv", index_col=False ).iloc[:,1:5]
#predictions_da2 = pd.read_csv("../results/mlpDA2_eval_preds_test_full.csv",index_col=False ).iloc[:,1:6]
#%%
data = [performance_mlp['test_mae'], performance_res['test_mae'], 
        performance_res2['test_mae'], performance_reg['test_mae']]

fig1, ax1 = plt.subplots(figsize=(8,8))
# ax1.set_title('MAE for temporal prediction')
ax1.boxplot(data)
ax1.set_xticklabels(["Naive\n Network", "Bias\n Correction", "Parallel\n Physics", "Physics\n Regularization"], fontsize=18)
ax1.set_ylabel("Mean absolute error [g C m$^{-2}$ day$^{-1}$]")
#%%
def plot_prediction(y_tests, predictions, mae=None, rmse=None, main=None):
    
    """
    Plot Model Prediction Error (root mean squared error).
    
    """
    
    fig, (ax1,ax2) = plt.subplots(2, figsize=(8,8), gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.02}, sharex=True)
    fig.subplots_adjust(wspace=0.2)
    ax2.axhline(y=0.0, xmin=0, xmax=365, color="black", linestyle="--", alpha=0.7)
    if main is not None:
        fig.suptitle(f'{main}', fontsize=20)
    
    ci_preds = np.quantile(np.array(predictions), (0.05,0.95), axis=1)
    m_preds = np.mean(np.array(predictions), axis=1)
    
    errors = np.subtract(predictions, y_tests)
    ci_errors = np.transpose(np.quantile(errors, (0.05,0.95), axis=1))
    m_errors = np.mean(errors, axis=1)

    ax1.fill_between(np.arange(len(ci_preds[0])), ci_preds[0],ci_preds[1], color="lightblue", alpha=0.9)
    ax1.plot(np.arange(len(y_tests)), y_tests, color="grey", label="$y$ observed", marker = "o", linewidth=0.7, alpha=0.9, markerfacecolor='lightgrey', markersize=4)
    ax1.plot(np.arange(len(m_preds)), m_preds, color="blue", label="$\hat{y}$ predicted", marker = "", alpha=0.5, linewidth=0.6)
    
    ax2.fill_between(np.arange(errors.shape[0]), ci_errors[:,0],ci_errors[:,1], color="lightsalmon", alpha=0.9)
    ax2.plot(m_errors, color="red", label="Error", marker = "", alpha=0.5)
    
    #except:
    #    print("Plotting Preles Predictions.")
    #    ax.plot(predictions, color="green", label="Predictions", marker = "", alpha=0.5)
    #    mae = metrics.mean_absolute_error(y_tests, predictions)
        
    fig.text(0.5, 0.02, "Day of Year", ha='center', size=20, family='Palatino Linotype')
    fig.text(0.00, 0.5, "Gross primary produdction [g C m$^{-2}$ day$^{-1}$]", va='center', rotation = 'vertical', size=20, family='Palatino Linotype')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        tick.label.set_fontfamily('Palatino Linotype') 
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        tick.label.set_fontfamily('Palatino Linotype') 
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(18) 
        tick.label.set_fontfamily('Palatino Linotype') 
    
    ax1.legend(loc="upper right", prop={'size':18, 'family':'Palatino Linotype'}, frameon=False)
    ax1.set_ylim((-2,17))
    
    

    if mae is not None:
        ax1.text(5, 15.0, f"MAE = {mae}", family='Palatino Linotype', size=18)
    if rmse is not None:
        ax1.text(5, 13.5, f"RMSE = {rmse}", family='Palatino Linotype', size=18)
    
#%%
def plot_running_losses(train_loss, val_loss, plot_train_loss, main=None,
                        colors=["blue", "lightblue"],
                        colors_test_loss = ["green","lightgreen"]):

    
    fig, ax = plt.subplots(figsize=(8,8))
    if main is not None:
        fig.suptitle(f'{main}', fontsize=20)

    if train_loss.shape[1] > 1:
        ci_train = np.quantile(train_loss, (0.05,0.95), axis=1)
        ci_val = np.quantile(val_loss, (0.05,0.95), axis=1)
        train_loss = np.mean(train_loss, axis=1)
        val_loss = np.mean(val_loss, axis=1)
        
        if plot_train_loss:
            ax.fill_between(np.arange(len(train_loss)), ci_train[0],ci_train[1], color=colors[1], alpha=0.3)
        ax.fill_between(np.arange(len(train_loss)), ci_val[0],ci_val[1], color=colors_test_loss[1], alpha=0.3)
    
    else: 
        train_loss = train_loss.reshape(-1,1)
        val_loss = val_loss.reshape(-1,1)
    
    if plot_train_loss:
        ax.plot(train_loss, color=colors[0], label="Training loss", linewidth=1.2)
        ax.plot(val_loss, color="green", label = "Test loss", linewidth=1.2)
    else:
        ax.plot(val_loss, color=colors_test_loss[0], label = "Test loss\nfull re-training", linewidth=1.2)
    #ax[1].plot(train_loss, color="green", linewidth=0.8)
    #ax[1].plot(val_loss, color="blue", linewidth=0.8)
    fig.text(0.0, 0.5, "Mean absolute error [g C m$^{-2}$ day$^{-1}$]",va='center', rotation= 'vertical', size=20)
    fig.text(0.5, 0.02, "Epochs", ha='center', size=20)
    #plt.ylim(bottom = 0.0)
    for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(18) 
    plt.rcParams.update({'font.size': 18})
    
    ax.legend(loc="upper right", frameon=False)
    
    
#%%
mae = np.mean(performance_mlp, axis=0)
print(mae, np.std(performance_mlp, axis=0))
plot_prediction(y2008, predictions_mlp, np.round(mae[3], 3), np.round(mae[2], 3), main="Naive Network")

plot_running_losses(tloss_mlp, vloss_mlp, main="Naive network", plot_train_loss=True)
#%%
mae = np.mean(performance_res, axis=0)
print(mae, np.std(performance_res, axis=0))
plot_prediction(y2008s, predictions_res, np.round(mae[3], 3), np.round(mae[2], 3), main="Bias correction")

plot_running_losses(tloss_res, vloss_res, plot_train_loss=True)
#%%
mae = np.mean(performance_reg, axis=0)
print(mae, np.std(performance_reg, axis=0))
plot_prediction(y2008, predictions_reg, np.round(mae[3], 3), np.round(mae[2], 3), main="Physics regularization")

plot_running_losses(tloss_reg, vloss_reg, plot_train_loss=True)
#%%
mae = np.mean(performance_res2, axis=0)
plot_prediction(y2008s, predictions_res2, np.round(mae[3], 3), np.round(mae[2], 3), main="Parallel physics")
print(mae, np.std(performance_res2, axis=0))

plot_running_losses(tloss_res2, vloss_res2, plot_train_loss=True)
#%%
mae = np.mean(performance_da1, axis=0)
plot_prediction(y2008, predictions_da1, np.round(mae[3], 3), np.round(mae[2], 3), main="Domain adaptation")

plot_running_losses(tloss_da1, vloss_da1, plot_train_loss=True)
#%%
mae = np.mean(performance_da1.iloc[1:,:], axis=0)
plot_prediction(y2008, predictions_da1.drop(['test_mlp1'], axis=1), np.round(mae[3], 3), np.round(mae[2], 3), main="Domain adaptation")

#%% Collect results from AS and HP search
res_as = pd.read_csv("../results/NresAS_full.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
print(list(b[np.isfinite(b)].astype(np.int)))

res_hp = pd.read_csv("../results/NresHP_full.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
print(b)

#%%
res_as = pd.read_csv("../results/NmlpAS_full.csv")
ass = res_as.loc[res_as.val_loss.idxmin()][1:5]
bs = ass.to_numpy()
print(list(bs[np.isfinite(bs)].astype(np.int)))

res_hp = pd.read_csv("../results/NmlpHP_full.csv")
ah = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
bh = ah.to_numpy()
print(bh)


#%%
res_as = pd.read_csv("../results/NregAS_full.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
print(list(b[np.isfinite(b)].astype(np.int)))

res_hp = pd.read_csv("../results/NregHP_full.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
print(b)

#%%
res_as = pd.read_csv("../results/NresAS2_full.csv")
a = res_as.loc[res_as.val_loss.idxmin()][1:5]
b = a.to_numpy()
print(list(b[np.isfinite(b)].astype(np.int)))

res_hp = pd.read_csv("../results/NresHP2_full.csv")
a = res_hp.loc[res_hp.val_loss.idxmin()][1:3]
b = a.to_numpy()
print(b)

#%%

a1 = [1,2,3]
a2 = [2,3,4]

print((np.array(a1)**2 + np.array(a2)**2)/2)