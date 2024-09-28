# Process-informed neural networks: a hybrid modelling approach to improve predictive performance and inference of neural networks in ecology and beyond

This repository contains the code that accompanies the article (DOI). The process-based model PRELES is combined with neural networks into process-informed neural networks (PINNs), which incorporate the process model knowledge into the neural network training algorithm and structure. Prediction tasks for C-fluxes in temperate forests are systematically evaluated with five different types of PINNs (i) in data-rich and data-sparse regimes (ii) in temporal, spatial and spatio-temporal prediction scenarios. <br/>

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

- R: The scripts run safely in R v4.3.1 / v4.2.1 with the package versions listed in requirements_r.txt. Packages can be installed with "install_packages.R", missing packages are installed on the fly when running main.R. 

- Python: Package versions and dependencies are listed in requirements.txt. Create a virtual environment before installation, e.g. directly in your local project directory with
```console
python -m venv pinns
source venv/bin/activate

pip install -r requirements.txt
```

- Preles-Compilation: For instructions on PRELES compilation, see README file in src folder.

## Computing instructions

1. Calibrate PRELES and generate predictions

2. Run neural architecture search

3. Pre-training for domain adaptation

4. Evaluate models

5. Variable importance analysis

