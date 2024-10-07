# Process-informed neural networks: a hybrid modelling approach to improve predictive performance and inference of neural networks in ecology and beyond

This repository contains the code that accompanies the article (DOI). The process-based model PRELES is combined with neural networks into what we call process-informed neural networks (PINNs), in accordance with similar approaches in physics. We broadly understand as PiNNs any type of neural network that incorporates process model knowledge during the training. PiNNs learn weights and biases jointly from process-model simulations and observations. Simulations from the process model, or the model structure itself constrain the training algorithm during optimisation. The trained PiNNs then perform prediction tasks for C-fluxes in temperate forests (specifically: Gross primary prodctivity, GPP), and they are evaluated and compared with the predictions of PRELES and a vanilla neural network. The provided code considers five different types of PINNs which are 

- bias correction NN (*res*)
- parallel process NN (*res2*)
- regularised NN (*ref*)
- pretrained NN (domain adapted) (*DA*)
- physics embedding NN (*emb*)

PINNs are evaluated (i) in temporal, spatial and spatio-temporal prediction scenarios (see folder structure) (i) and in data-rich and data-sparse regimes. The instructions below will walk you through the code if you want to reproduce our or create your own PINNs. Because of heavy ressource use during PiNN development, **we strongly recommend you to work on an HPC with the below instructions** and submit them stepwise with bash scripts (for example, see my_bash_script.sh) to your scheduling system. This will be specifically required during the neural architecture search and the model training. <br/>

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

- R: The scripts run safely in R v4.3.1 / v4.2.1 with the package versions in r/requirements.txt. For their manual installation, run "install_packages.R". Alternatively, make sure to have R loaded if you're working on your HPC. Navigate to r and run main.R with Rscript. This will install the renv package, if not allready installed, which will be used to restore the project from the lockfile in R. It installs missing package versions and dependencies specified in r/renv and afterwards conduct the analysis as specified in r/README.

```console
@PiNNs~:cd r
@PiNNs/r~:Rscript main.R
```

- Python: Package versions and dependencies are in environment.yml. Make sure you have conda module loaded and create the conda environment from environment.yml

```console
@PiNNs~:conda env create -f environment.yml
@PiNNs~:conda activate pinns
```
After initializing the environment it is essential to create the PRELES pytorch c++ extension. Please proceed in the folder *src* and follow the README file.

## Data, Models and Hyperparameters

The data, models and hyperparamters we used for our experiments are available at in an OSF repository https://osf.io/7gzbn/ (DOI: 10.17605/OSF.IO/7GZBN). Please cite our publication if you use them.

## PRELES calibration and prediction

If you want to use our data files, you can skip these steps and download the four listed files from the OSF repository. Otherwise: For the PRELES calibration and simulation, run main.R (in *r*). This will call PRELES_predictions.R and conditional_via.R. In PRELES_predictions.R, we fit PRELES with BayesianTools (https://github.com/florianhartig/BayesianTools) in, make GPP predictions in all model scenarios, and evaluate them against observations. For inference, we run a conditional variable importance analysis in conditional_via.R. **Attention**: PRELES_predictions.R with the default sampling of 50000 per MCMC chain will take hours to run! If you want to conduct a test run, reduce sample size strongly. Running main.R will create a *results* subfolder in each predictive experiment folder. PRELES performances are written to .csv files and saved in *results*, whereas the neural network input data files, merged with PRELES predictions, are saved to *data*, **if your flag save_data = TRUE**.

We create four files in total. hyytiala files will be loaded with the temporal prediction scenario, allsites files will be loaded with the spatial and spatio_temoporal prediction scenario. The endings _sparse and _full indicate the data availablity scenario. 

- hyytialaF_full.csv
- hyytialaF_sparse.csv
- allsitesF_full.csv
- allsitesF_sparse.csv

The following four steps to develop the PiNNs are carried out in most parts for each prediction scenario separately, with overlaps in the spatial and spatio-temporal experiment.

## Neural achitecture search

Now that we have the neural network input data, we can start a neural architecture and hyperparameter search (NAS) to optimise PINN and algorithmic parameters. We run a sparate NAS for purely temporal (*temporal/nas*) and for spatial and spatio-temporal (*spatial/nas*) PINNs. Each *nas* folder contains four scripts, the endings of which indicate the target PiNN for the respective architecture: mlp, reg, res, res2 (see above). The architecture for the emb was constructed and tuned manually, for the DA we reused the mlp architecture.

Before proceeding with the architecture and hyperparameter search create a results folder as follows
```console
@PiNNs~:cd temporal/nas
@PiNNs~:mkdir results
```

Now execute the following lines of code. Note that the flag -d is used to specify the data scenario (full or sparse).
```console 
@PiNNs~:python ENmlp.py -d full
@PiNNs~:python ENmlp.py -d sparse
```

You can reduce runtime by reducing the search space of the random search with parameters **agrid** and **pgrid** in *misc/NASSearchSpace*. To run the network architecture and hyperparameter search for the spatial and spatio-temporal experiments follow the same steps in the folder *spatial*, e.g.
```console
@PiNNs~: cd spatial/nas
@PiNNs~: mkdir results
@PiNNs~: python EN2mlp.py -d full
```


## Pretraining

Before neural network training, one model version is pretrained on PRELES simulations (in *pretraining*). To prepare the data set for the pre-training, run *simulation.py*.
Assuming you are in the main folder PiNNs and want to proceed with the temporal experiment, run
```console
@PiNNs~:cd temporal/pretraining
@PiNNs~:mkdir results
@PiNNs~:python simulations.py -d full -n 5000
```
where the flag -d defines the data scenario an -n the parameter sample size. This file calls the respective data file, e.g. hyttialaF_full.csv. Running the simulation will create a large synthetic data set: (1) input data is augmented with Generalised Additive Models, (2) PRELES parameters are sampled at a predefined size in a Latin Hypercube Design and (3) PRELES GPP simulations are generated. This file will be saved to the experiments' respective results folder. 

Next call pretraining.py. This script will use the vanilla MLP architecture (*results/NmlpHP_{data_use}.csv*, data_use = full or sparse) to train on the synthetic data set. Make sure to run this on your HPC node, as specifically for the spatial scenario runtime is extensive. To reduce runtime, change parameter sample size with the flag -n. Proceed as
```console
@PiNNs~:cd temporal/pretraining/
@PiNNs~:python pretraining.py -d full -n 5000
```

## Neural network training and PiNN predictions

Training and evaluation happen simulatenously when calling the model-specific evaluation files in respective *analysis* folders. Each run calls *training.py* from *misc* and requires model specific files for architectural choices (as e.g. in *nas/results/NmlpHP_{data_use}.csv*). Change hard-coded epochs in the eval_function if you want to reduce runtime. Again, set data availability scenario manually with the flag -d full or -d sparse. Before running the training and evaluation scripts create a results folder. Proceed e.g. as
```console
@PiNNs~:cd temporal/analysis
@PiNNs~:mkdir results
@PiNNs~:python evalmlp.py -d full
```
Model performances will be saved to the respective results folder, while model parameters will be saved to the folder *temporal/models*. Note that for the domain adaptation the parameter *N* in evalmlpDA.py must be the same as the flag -n in simulations.py and pretraining.py. Further note that eval2mlpDA.py requires the flag -a which we set in our experiments to 1. It defines which layers should be retrained with the empirical data, where -a 1 defines all layers to be retrained and -a 2 defines only last layers to be retrained.


## Variable importance analysis

Now that PiNNs are trained and evaluated, first inference steps are conducted with a conditional variable importance analysis. In each *analysis* folder, call *via_conditional.py*. This file requires the model structure selected in the NAS, taken from *NmlpHP_{data_use}.csv*. It also requires model state dictionaries (trained network weights and biases), saved during model training to e.g. *temporal/models*. The via_conditional.py script will analyse variable importances for each input variable separately at the seasonal mean of the other variables. The flag -m defines the model to be analysed and the flag -d defines the data scenario. Proceed as
```console
@PiNNs~:cd temporal/analysis
@PiNNs~:python via_conditional.py -m mlp -d full
```

## Visualise results

Finally, to visualise results and save plots of comparitive model performances and variable importances, created as described above, call *misc/visualise_results.py*.
