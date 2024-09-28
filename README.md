# Process-informed neural networks: a hybrid modelling approach to improve predictive performance and inference of neural networks in ecology and beyond

This repository contains the code that accompanies the article (DOI). The process-based model PRELES is combined with neural networks into what we call process-informed neural networks (PINNs), in accordance with similar approaches in physics. We broadly understand as PiNNs any type of neural network that incorporates process model knowledge during the training. PiNNs learn weights and biases jointly from process-model simulations and observations. Simulations from the process model, or the model structure itself constrain the training algorithm during optimisation. The trained PiNNs then perform prediction tasks for C-fluxes in temperate forests (specifically: Gross primary prodctivity, GPP), and they are evaluated and compared with the predictions of PRELES and a vanilla neural network. The provided code considers five different types of PINNs which are 

- bias correction NN (*res*)
- parallel process NN (*res2*)
- regularised NN (*ref*)
- pretrained NN (domain adapted) (*DA*)
- physics embedding NN (*emb*)

PINNs are evaluated (i) in temporal, spatial and spatio-temporal prediction scenarios (see folder structure) (i) and in data-rich and data-sparse regimes. The instructions below will walk you through the code if you want to reproduce our or create your own PINNs. <br/>

The repository is structured as follows:
- ./src: contains the source code of the model PRELES
- ./data: contains the data used to force PRELES and the neural networks (for the pre-processed version, see OSF repository: DOI 10.17605/OSF.IO/7GZBN)
- ./r: contains the R scripts used for PRELES calibration and variable importance analyses
- ./misc: contains the scripts shared among prediction experiments
- ./spatial, ./temporal, ./spatio-temporal: contain the scripts used to run the experiments at three different prediction scenarios

Data source for creating PRELES and the neural networks forcing can be found at: https://github.com/COST-FP1304-PROFOUND/ProfoundData<br/>
C source code of the model PRELES in ./src can be found at: https://github.com/MikkoPeltoniemi/Rpreles<br/>

## Set up R and Python environments

PRELES calibration is conducted in R with the BayesionTools package, while integration with, neural network training and evaluation is conducted in Python. Therefore we need to setup both computing environments before results can be reproduced. In addition, we compile a python version of PRELES for reproducing results of the Physics Embedding. 

- R: The scripts run safely in R v4.3.1 / v4.2.1 with the package versions in requirements_r.txt. For installation, run "install_packages.R", or directly run main.R where packages will be installed on the fly. 

- Python: Package versions and dependencies are in requirements.txt. Create a virtual environment with before installation, e.g. directly in your project directory with

```console
@PiNNs~:python -m venv pinns
@PiNNs~:source venv/bin/activate

@PiNNs~:pip install -r requirements.txt
```
We will need a compiled version of PRELES for creating the embedded physics NN: see the accompanying README file in *src* folder.

## Data, Models and Hyperparameters

The data, models and hyperparamters we used for our experiments are available at in an OSF repository https://osf.io/7gzbn/ (DOI: 10.17605/OSF.IO/7GZBN). If you use them, we are always happy about a citation.

## PRELES calibration and prediction

For the PRELES calibration and simulation, run main.R (in *r*). This will call install_packages.R, PRELES_predictions.R and conditional_via.R. In PRELES_predictions.R, we fit PRELES with BayesianTools (https://github.com/florianhartig/BayesianTools) in, make GPP predictions in all model scenarios, and evaluate them against observations. For inference, we run a conditional variable importance analysis in conditional_via.R. **Attention**: PRELES_predictions.R with the default sampling of 50000 per MCMC chain will take hours to run! If you want to conduct a test run, reduce sample size strongly. Running main.R will create a *results* subfolder in each predictive experiment folder. PRELES performances are written to .csv files and saved in *results*, whereas the neural network input data files, merged with PRELES predictions, are saved to *data*.

## Neural achitecture search

## Neural network training

## PiNN predictions

## Variable importance analysis


